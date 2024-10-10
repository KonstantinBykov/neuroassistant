### В данном проекте я создал нейропомощника по первой медицинской помощи используя такую LLM как saiga_mistral_7b, а в качестве базы знаний взял графы знаний.
    from huggingface_hub import login
    HF_TOKEN="***************************"
    #Вставьте ваш токен (здесь указан временный токен)
    login(HF_TOKEN, add_to_git_credential=True)

    def messages_to_prompt(messages):
        prompt = ""
        for message in messages:
            if message.role == 'system':
                prompt += f"{message.role}\n{message.content}\n"
            elif message.role == 'user':
                prompt += f"{message.role}\n{message.content}\n"
            elif message.role == 'bot':
                prompt += f"bot\n"
    
        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("system\n"):
            prompt = "system\n\n" + prompt
    
        # add final assistant prompt
        prompt = prompt + "bot\n"
        return prompt

    def completion_to_prompt(completion):
        return f"system\n\nuser\n{completion}\nbot\n"

### Загрузка модели

    #Определяем параметры квантования, иначе модель не выполниться в колабе
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    #Задаем имя модели
    MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
    
    #Создание конфига, соответствующего методу PEFT (в нашем случае LoRA)
    config = PeftConfig.from_pretrained(MODEL_NAME)
    
    #Загружаем базовую модель, ее имя берем из конфига для LoRA
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,          # идентификатор модели
        quantization_config=quantization_config, # параметры квантования
        torch_dtype=torch.float16,               # тип данных
        device_map="auto"                        # автоматический выбор типа устройства
    )
    
    #Загружаем LoRA модель
    model = PeftModel.from_pretrained(
        model,
        MODEL_NAME,
        torch_dtype=torch.float16
    )
    
    #Переводим модель в режим инференса
    #Можно не переводить, но явное всегда лучше неявного
    model.eval()
    
    #Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

### Загрузка модели во фреймворк LlamaIndex

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    print(generation_config)
    
    llm = HuggingFaceLLM(
        model=model,             # модель
        model_name=MODEL_NAME,   # идентификатор модели
        tokenizer=tokenizer,     # токенизатор
        max_new_tokens=generation_config.max_new_tokens, # параметр необходимо использовать здесь, и не использовать в generate_kwargs, иначе ошибка двойного использования
        model_kwargs={"quantization_config": quantization_config}, # параметры квантования
        generate_kwargs = {   # параметры для инференса
          "bos_token_id": generation_config.bos_token_id, # токен начала последовательности
          "eos_token_id": generation_config.eos_token_id, # токен окончания последовательности
          "pad_token_id": generation_config.pad_token_id, # токен пакетной обработки (указывает, что последовательность ещё не завершена)
          "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
          "repetition_penalty": generation_config.repetition_penalty,
          "temperature": generation_config.temperature,
          "do_sample": True,
          "top_k": 50,
          "top_p": 0.95
        },
        messages_to_prompt=messages_to_prompt,     # функция для преобразования сообщений к внутреннему формату
        completion_to_prompt=completion_to_prompt, # функции для генерации текста
        device_map="auto",                         # автоматически определять устройство
    )

### Воспользуемся документом с правилами первой медицинской помощи

    documents = SimpleDirectoryReader("./data").load_data()
    
    from langchain_huggingface  import HuggingFaceEmbeddings
    embed_model = LangchainEmbedding(
      HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )

### Настраиваем окружение для LlamaIndex:

    #Настройка ServiceContext (глобальная настройка параметров LLM)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    
    #Создаем простое графовое хранилище
    graph_store = SimpleGraphStore()
    
    #Устанавливаем информацию о хранилище в StorageContext
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    
    
    #Запускаем генерацию индексов из документа с помощью KnowlegeGraphIndex
    indexKG = KnowledgeGraphIndex.from_documents( documents=documents,               # данные для построения графов
                                               max_triplets_per_chunk=3,        # сколько обработывать триплетов связей для каждого блока данных
                                               show_progress=True,              # показывать процесс выполнения
                                               include_embeddings=True,         # включение векторных вложений в индекс для расширенной аналитики
                                               storage_context=storage_context) # куда сохранять результаты

### Визуализация графов


    g = indexKG.get_networkx_graph(500)
    net = Network(notebook=True,cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.show("graph.html")
    net.save_graph("Knowledge_graph.html")
    
    IPython.display.HTML(filename="/content/Knowledge_graph.html")
    
    !unzip -qo "storage.zip" -d ./storage
    
    #устанавливаем соответствия
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage"),
        graph_store=SimpleGraphStore.from_persist_dir(persist_dir="./storage"),
        vector_store=SimpleVectorStore.from_persist_dir(
            persist_dir="./storage"
        ),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage"),
    )
    
    from llama_index.core import (
        load_index_from_storage,
        load_indices_from_storage,
        load_graph_from_storage,
    )
    #загружаем данные
    indexKG = load_index_from_storage(storage_context)

#### Сделаем запрос к нашей модели

    #Список ключевых слов и фраз
    keywords = ["похоть ", "чревоугодие ", "гордыня", "уныние", "жадность", "гнев", "зависть"]
    
    query = "Какие есть способы проверки дыхания?"
    query_engine = indexKG.as_query_engine(include_text=True, verbose=True)
    
    
    
    def filter_query(query):
        # Приводим запрос к нижнему регистру для упрощения проверки
        query_lower = query.lower()
    
        # Проверяем, содержит ли запрос одно из ключевых слов
        for keyword in keywords:
            if keyword in query_lower:
                return False
        return True
    
    def process_query(query):
        if filter_query(query):
            # Формируем сообщение для query_engine
            message_template = f"""system
            Вы врач скорой помощи. Отвечайте согласно Источнику. Проверьте, содержит ли Источник ссылки на ключевые слова Вопроса.
            Если нет, просто скажите «Я не знаю». Не выдумывайте! 
            user
            Вопрос: {query}
            Источник:
            
            """
            # Выполняем запрос
            response = query_engine.query(message_template)
            return response.response
        else:
            return "Я не знаю"

### Пример использования
    response = process_query(query)
    print('Ответ:')
    print(response)

    Extracted keywords: ['способы', 'проверки', 'дыхания']
    KG context:
    The following are knowledge sequence in max depth 2 in the form of directed graph like:
    `subject -[predicate]->, object, <-[predicate_next_hop]-, object_next_hop ...`
    ('Сделать два вдоха искусственного дыхания', 'Is', 'Выполняются следующим образом')
    ('Необходимо', 'Is', 'Осуществлять искусственное дыхание методом «рот-ко-рту»')
    Ответ:
    Есть несколько способов проверки дыхания:
    1. Наблюдать за движениями грудной клетки: если она поднимается и опускается, значит, человек дышит.
    2. Наблюдать за движениями живота: если живот поднимается и опускается, значит, человек также дышит.
    3. Наблюдать за движениями рта: если рт поднимается и опускается, значит, человек тоже дышит.
    4. Нажать на грудь: если грудь поднимается и опускается, значит, человек, возможно, дышит.
    5. Использовать средства для фиксации шейного отдела позвоночника: они могут быть использованы элементами одежды (курка, свитер и т. п.), которые оборачивают вокруг шеи, предотвращая сдавление мягких тканей и органов шеи, но добиваясь того, чтобы края импровизированного воротника туго подпирали голову.
    6. Сделать два вдоха искусственного дыхателя: это может быть выполнено методом "рот-ко-рту".

### Результат отличный! Обратите внимание на вывод модели. Она извлекает ключевые слова, по которым и осуществляется поиск в базе знаний.
