# RAG & IRQA Шаблон за чатбот с въпроси и отговори по природни науки

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Chainlit](https://img.shields.io/badge/UI-Chainlit-F75591?logo=chainlit&logoColor=white)
![RAG](https://img.shields.io/badge/Tech-RAG%20%26%20IRQA-orange)

> **Реален случай на употреба и свързана дисертация**
> * **Вижте реалния случай на употреба**: [https://acer2.snu.ac.kr](https://acer2.snu.ac.kr) (в реална употреба от 2020 г.)
> * **Свързана дисертация**: [Линк към дисертацията](https://s-space.snu.ac.kr/handle/10371/222093?mode=full)


Този проект е шаблон с отворен код за образователен чатбот по природни науки, комбиниращ технологиите **RAG (Retrieval-Augmented Generation / Генерация с разширено извличане)** и **IRQA (Information Retrieval-based Question Answering / Въпроси и отговори на базата на извличане на информация)**.
Базиран на Chainlit, той едновременно предоставя на потребителя **отговор от генеративен ИИ** и **най-сходния въпрос-отговор от базата данни**.

Проектът е създаден така, че други разработчици да могат да го персонализират, като заменят данните с техни собствени (материали с въпроси и отговори по други предмети, по други области, съдържание на PDF файлове и др.).


## Основни характеристики (Key Features)

* **Система, комбинираща LLM и съществуващ набор от данни с въпроси и отговори**:
    * **LLM**: Локален LLM, управляван от Google Gemma, Gemini или Ollama, разбира контекста и обяснява любезно.
    * **Предлагане на сходни съществуващи въпроси и отговори**: Търси в векторната БД съществуващите Q&A (въпроси и отговори на ученици, училищна информация и др.), най-сходни с въпроса, и показва оригиналните данни заедно с отговора. По този начин може да се осъществи кръстосана проверка срещу халюцинации на ИИ.
* **Гъвкав избор на LLM**: Може да избирате между облачен (Google Gemini) API и локален (Ollama) модел за LLM.
* **Мултимодална поддръжка**: Можете да качвате не само текст, но и изображения и PDF файлове за анализ и задаване на въпроси.
* **Автоматично изграждане на векторно хранилище**: Поставете xlsx или PDF файл и стартирайте скрипта — FAISS векторната БД се изгражда автоматично.
* **Запис на удовлетвореността на потребителите**: Съхранява обратната връзка за удовлетвореността на потребителите в MariaDB (БД с отворен код) за бъдещо подобряване на модела.

## Технологичен стек

* **Фронтенд**: [Chainlit](https://docs.chainlit.io)
* **LLM**: Google Gemini или Gemma (Cloud), локален LLM, управляван от Ollama (Local)
* **RAG / векторно хранилище**: FAISS (CPU based), LangChain
* **База данни**: MariaDB (съхранение на записи за употреба и оценки от потребители)
* **Обработка на естествен език**: Kiwi (морфологичен анализ на корейски), HuggingFace Embeddings

---


## 💻 Инсталация и стартиране (Getting Started)


### 1. Предварителни изисквания
* Операционна система: Linux
* Python: 3.13 или по-нова версия
* При използване на LLM с отворен код — инсталиране на [Ollama](https://ollama.com/) и изтегляне на модел (препоръчва се `EXAONE 3.5 7.8B` на LG AI Research или `gemma3:12b-it-qat` на Google или по-висок)
* База данни: MariaDB


### 2. Инсталация

* Необходими са инсталирани git и uv. Изтеглете шаблона от Git и създайте виртуална среда.

```bash
git clone https://github.com/kungmo/science_qna_rag_irqa_chatbot.git
cd science_qna_rag_irqa_chatbot
uv init --python 3.13
uv venv
```

* Сега влезте във виртуалната среда.

За Windows: ```./.venv/Scripts/activate```

За Linux или MacOS: ```source ./.venv/bin/activate```

* Инсталирайте Python пакетите, необходими за работата на чатбота.
```
uv pip install -r requirements.txt
```

* Инсталирайте Python пакетите за обработка на изображения.

За Windows: ```pip install python-magic-bin```

За Linux: ```apt install libmagic``` (за Ubuntu/Debian)

За MacOS: ```brew install libmagic``` (изисква инсталиран homebrew)

### 3. Настройка на променливи на средата (.env)
Създайте файл .env в корена на проекта.

```bash
GOOGLE_API_KEY=въведете_тук_Gemini_API_ключа
DB_HOST=127.0.0.1
DB_USER=въведете_тук_потребителско_име
DB_PASSWORD=въведете_тук_парола
DB_NAME=въведете_тук_името_на_БД
```


### 4. Създаване на таблици в базата данни (SQL)

* Отворете MariaDB базата данни чрез DBeaver и изпълнете следния SQL, за да създадете таблиците. (Задължително)
Приема се, че името на БД е chatbot_logs, а името на таблицата е datalog.

```sql
CREATE DATABASE IF NOT EXISTS `chatbot_logs` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE `chatbot_logs`;
CREATE TABLE `datalog` (
  `primarykey` int(11) NOT NULL AUTO_INCREMENT,
  `input_time` datetime NOT NULL DEFAULT current_timestamp(),
  `user_name` varchar(15) NOT NULL,
  `session_id` varchar(36) NOT NULL,
  `student_questio` varchar(2048) DEFAULT NULL,
  `answer` longtext DEFAULT NULL,
  `accuracy` smallint(5) unsigned DEFAULT NULL,
  `satisfaction` smallint(5) unsigned DEFAULT NULL,
  `image_path` varchar(255) DEFAULT NULL,
  `pdf_path` varchar(255) DEFAULT NULL,
  `selected_similar` float DEFAULT NULL,
  PRIMARY KEY (`primarykey`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
```

### 5. Изграждане на векторна БД (добавяне на собствени данни)

За да адаптирате чатбота за **вашите цели (въпроси и отговори по природни науки, училищни въпроси и отговори, или други)**, заменете данните.

    Папка data/:
    - Поставете Excel файлове с двойки въпрос-отговор по природни науки (df_qna_*.xlsx). (Колони: въпрос, отговор)
    - Поставете Excel файлове с двойки въпрос-отговор извън природните науки (df_cus_*.xlsx). (Колони: въпрос, отговор)

    Папка pdfs/:
    - Поставете PDF документи за справка.
    - Няколко PDF файла се обработват наведнъж.

    Стартирайте скриптовете за индексиране:

```bash
python rag_faiss_creator.py           # Индексиране на данни от Excel
python rag_faiss_creator_renew_pdf.py # Индексиране на данни от PDF
```

### 6. Стартиране

```Bash
chainlit run chainlit_memory.py
```


### Допълнително...

Тази рамка на чатбота е универсална и може да се използва в области извън въпросите и отговорите по природни науки.

### Лиценз

Apache License 2.0
