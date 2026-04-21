import chainlit as cl
from chainlit.config import config
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from pathlib import Path
import time
import datetime
import pytz
import base64
import asyncio
from typing import Dict, Any, List, Optional
import pymysql.cursors
import tempfile
import shutil
from PIL import Image, UnidentifiedImageError
import io
import magic
import pillow_heif
import json
import opendataloader_pdf
from mcp import ClientSession

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 환경 변수 설정
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
IMAGES_DIR = os.path.join(BASE_DIR, "uploaded_images")
PDFS_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
PDF_TEMP_DIR = os.path.join(BASE_DIR, "pdf_temp")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PDFS_DIR, exist_ok=True)
os.makedirs(PDF_TEMP_DIR, exist_ok=True)

# 여기서 LLM을 선택할 수 있습니다. 모두 바꾸기로 llm_gemini와 llm_local_server를 전환하세요.
llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=1.0)
#llm_gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=1.0, thinking_level="high")
#llm_local_server = ChatOllama(model="exaone3.5:7.8b-instruct-q8_0", temperature=0)
# ollama를 쓰려면 이 코드에 있는 llm_gemini 변수명를 llm_local_server로 모두 변경해야 함.
# 오픈소스 LLM 사용 시 반드시 Instruction Tuned 모델을 사용해야 함. 예) gemma3:12b-it-qat

def format_qna_docs(docs: List[Document]) -> List[str]:
    """QNA 문서를 포매팅"""
    formatted = []
    for doc in docs:
        if doc.metadata.get("Отговор"):
            qna = f"Въпрос: {doc.page_content}\nОтговор: {doc.metadata['Отговор']}"
            formatted.append(qna)
    return formatted


def format_pdf_docs(docs: List[Document]) -> List[str]:
    """PDF 문서를 포매팅"""
    formatted = []
    for doc in docs:
        content = f"Източник: {doc.metadata.get('source', 'Учебник')}\nСъдържание: {doc.page_content}"
        formatted.append(content)
    return formatted


def create_context_messages(qna_docs: List[Document] = [], pdf_docs: List[Document] = [],
                            uploaded_pdf_docs: List[Document] = None) -> List[BaseMessage]:
    """QNA와 PDF 문서를 별도의 시스템 메시지로 변환"""
    context_messages = []

    # 방금 업로드된 PDF 내용이 있으면 추가 (우선순위 1순위)
    if uploaded_pdf_docs and len(uploaded_pdf_docs) > 0:
        uploaded_pdf_context = "Текст, извлечен от PDF файла, качен от ученика:\n\n"
        for doc in uploaded_pdf_docs:
            chunk_text = f"Източник: {doc.metadata.get('source', 'PDF файл, качен от ученика')}\nСъдържание: {doc.page_content}"
            uploaded_pdf_context += chunk_text + "\n\n"
        context_messages.append(SystemMessage(content=uploaded_pdf_context))

    if qna_docs and len(qna_docs) > 0:
        qna_formatted = format_qna_docs(qna_docs)
        qna_context = "Свързани предишни въпроси и отговори:\n\n" + "\n\n".join(qna_formatted)
        context_messages.append(SystemMessage(content=qna_context))

    if pdf_docs and len(pdf_docs) > 0:
        pdf_formatted = format_pdf_docs(pdf_docs)
        pdf_context = "Материал, свързан с учебника:\n\n" + "\n\n".join(pdf_formatted)
        context_messages.append(SystemMessage(content=pdf_context))

    return context_messages


# 이미지 변환 함수
def convert_image_to_jpeg(image_path):
    """
    다양한 이미지 형식을 JPEG로 변환 (HEIC 지원 강화 및 16bit 이미지 대응)
    """
    try:
        # 파일 타입 감지
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(image_path)
        file_type_lower = file_type.lower()

        # [디버그] 파일 정보 터미널 출력
        print(f"[Debug] Converting image: {image_path} (MIME: {file_type})")

        img = None

        # HEIC/HEIF 파일인 경우 (확장자 또는 MIME 타입으로 확인)
        if ("heic" in file_type_lower or "heif" in file_type_lower) or image_path.lower().endswith(('.heic', '.heif')):
            # pillow_heif를 사용하여 명시적으로 로드 (HDR -> 8bit 변환)
            heif_file = pillow_heif.open_heif(image_path, convert_hdr_to_8bit=True)
            img = heif_file.to_pillow()
        else:
            # 일반 이미지는 바로 열기
            img = Image.open(image_path)

        # [핵심 수정] 이미지 모드 변환 (I;16, LAB, P, RGBA 등을 RGB로 강제 변환)
        # JPEG는 16비트나 투명도(Alpha)를 지원하지 않으므로 변환이 필수입니다.
        if img.mode in ('RGBA', 'P', 'LA', 'I', 'I;16', 'L'):
            img = img.convert('RGB')

        # 메모리 버퍼에 JPEG로 저장
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        # base64 인코딩
        jpeg_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jpeg_base64, None
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        # [디버그] 오류 발생 시 터미널에 빨간색(혹은 눈에 띄게) 출력
        print(f"\n❌ [ERROR] convert_image_to_jpeg failed:\n{traceback_str}\n")
        return None, f"Възникна грешка при конвертиране на изображение.: {str(e)}\n{traceback_str}"


def save_image_file(image_path, user_name):
    try:
        # 파일 타입 감지
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(image_path)
        file_type_lower = file_type.lower()

        # 타임스탬프 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_username = user_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        # HEIC/HEIF 파일인 경우 JPEG로 변환하여 저장
        if ("heic" in file_type_lower or "heif" in file_type_lower) or image_path.lower().endswith(('.heic', '.heif')):
            # 직접 로드 및 8bit 변환
            heif_file = pillow_heif.open_heif(image_path, convert_hdr_to_8bit=True)
            img = heif_file.to_pillow()

            # JPEG로 저장 파일명 설정
            new_filename = f"{timestamp}_{sanitized_username}.jpg"
            save_path = os.path.join(IMAGES_DIR, new_filename)

            # [핵심 수정] 16비트 등을 8비트 RGB로 변환
            if img.mode in ('RGBA', 'P', 'LA', 'I', 'I;16', 'L'):
                img = img.convert('RGB')

            img.save(save_path, format="JPEG")
        else:
            # 기타 이미지 파일은 원래 확장자 유지
            original_extension = Path(image_path).suffix
            if not original_extension:
                if "jpeg" in file_type_lower or "jpg" in file_type_lower:
                    original_extension = ".jpg"
                elif "png" in file_type_lower:
                    original_extension = ".png"
                else:
                    original_extension = ".jpg"

            extension = original_extension
            new_filename = f"{timestamp}_{sanitized_username}{extension}"
            save_path = os.path.join(IMAGES_DIR, new_filename)
            shutil.copy2(image_path, save_path)

        return save_path, None
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        # [디버그] 터미널 출력
        print(f"\n❌ [ERROR] save_image_file failed:\n{traceback_str}\n")
        return None, f"Възникна грешка при запазване на изображението.: {str(e)}\n{traceback_str}"


def save_pdf_file(pdf_path, user_name):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_username = user_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        new_filename = f"{timestamp}_{sanitized_username}.pdf"
        save_path = os.path.join(PDFS_DIR, new_filename)
        shutil.copy2(pdf_path, save_path)
        return save_path, None
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return None, f"Възникна грешка при запазване на PDF файла.: {str(e)}\n{traceback_str}"


def extract_text_from_pdf_with_opendataloader(pdf_path, pdf_filename):
    """opendataloader_pdf를 사용하여 한글 PDF 텍스트 추출

    사용자가 업로드한 PDF는 간단한 처리로 진행 (문단 길이 필터링 없음)
    """
    try:
        temp_output_dir = os.path.join(PDF_TEMP_DIR, f"temp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_output_dir, exist_ok=True)

        # PDF를 JSON으로 변환
        opendataloader_pdf.convert(
            input_path=[pdf_path],
            output_dir=temp_output_dir,
            format=["json"]
        )

        # 변환된 JSON 파일 찾기
        json_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.json')]
        if not json_files:
            return None, "JSON 파일이 생성되지 않았습니다."

        json_file_path = os.path.join(temp_output_dir, json_files[0])

        # JSON 파일 읽기
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 텍스트 추출 (간단한 처리)
        documents = []

        if 'kids' not in data:
            return None, "PDF 파일에 내용이 들어 있지 않습니다."

        current_heading = ""
        current_content_parts = []

        for kid in data['kids']:
            if kid['type'] == 'heading':
                # 이전 내용이 있으면 저장
                if current_content_parts:
                    full_content = current_heading + "\n" + " ".join(current_content_parts)
                    if len(full_content.replace(" ", "")) >= 10:
                        doc = Document(
                            page_content=full_content,
                            metadata={"source": pdf_filename}
                        )
                        documents.append(doc)

                current_heading = kid['content']
                current_content_parts = []

            elif kid['type'] == 'paragraph':
                current_content_parts.append(kid['content'])

        # 마지막 내용 저장
        if current_content_parts:
            full_content = current_heading + "\n" + " ".join(current_content_parts)
            if len(full_content.replace(" ", "")) >= 10:
                doc = Document(
                    page_content=full_content,
                    metadata={"source": pdf_filename}
                )
                documents.append(doc)

        # 추출된 청크가 길면 추가로 분할
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            split_documents = []
            for doc in documents:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    split_documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": pdf_filename}
                        )
                    )
            documents = split_documents

        # 임시 디렉토리 정리
        shutil.rmtree(temp_output_dir, ignore_errors=True)

        if not documents:
            return None, "PDF에서 추출된 텍스트가 없습니다."

        return documents, None

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        # 임시 디렉토리 정리
        if 'temp_output_dir' in locals():
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        return None, f"PDF 텍스트 추출 중 오류 발생: {str(e)}\n{traceback_str}"


def create_dynamic_system_prompt(
        user_name: str = "Нерегистриран",
        use_pdf_only: bool = False,
        uploaded_pdf_docs: Optional[List[Document]] = None
) -> SystemMessage:
    """
    사용자 이름과 현재 시각을 XML 구조의 컨텍스트로 포함하여 시스템 프롬프트 생성
    """

    # 1. 현재 날짜 및 시각 처리 (한국 시간)
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.datetime.now(kst)
    weekday_kor = ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]
    current_time_str = now.strftime(f"%Y년 %m월 %d일 {weekday_kor}요일 %H시 %M분")

    # 2. XML 컨텍스트 구성 (들여쓰기는 가독성을 위함)
    # 사용자 이름이 있을 때만 태그를 생성하거나, 'Нерегистриран'이라도 태그를 유지할지 결정합니다.
    # 여기서는 이름이 있을 때만 태그를 넣어 명확성을 높였습니다.
    user_tag = ""
    if user_name and user_name != "Нерегистриран":
        user_tag = f"    <user_name>{user_name}</user_name>\n"

    # 상단에 위치할 메타 정보 블록 생성
    metadata_xml = f"""<metadata>
    <current_time>{current_time_str}</current_time>
{user_tag}</metadata>"""

    # 3. 공통 프롬프트 내용 정의
    # 기존 코드에서 {name_context} 부분이 사라지고, 대신 상단의 <context>를 참조하게 됩니다.

    # PDF가 있는 경우
    if use_pdf_only and uploaded_pdf_docs:
        return SystemMessage(content=f"""{metadata_xml}
<task>Предоставете точен и ясен отговор на въпроса на потребителя в рамките на 200 думи.</task>
<format_instructions>
    1. Извеждайте в Markdown формат
    2. При писане на формули стриктно спазвайте нотацията на LaTeX
        <good_example>
            (1) $$ \\alpha = \\frac{{\\text{{числител}}}}{{\\text{{знаменател}}}} \\times 100% $$ 
            (2) $$ \\text{{Ag}}^+ $$
        </good_example>
    3. Представяйте URL адресите като обикновен текст (plain text)
</format_instructions>
<output_instructions>
    1. Разделете отговора на две части: „Мисли“ и „Отговор“
    2. В раздела „Мисли“ напишете следното подред:
        - Изисквания: Обобщение на ключовите точки, изисквани във въпроса
        - Основания и цитати: Подреждане на основания и потенциални цитати, които могат да бъдат използвани в отговора
        - Оценка: Оценка на уместността на всяко основание и определяне на посоката на окончателния отговор
    3. Отговорете въз основа на текстовото съдържание, извлечено от PDF файла, качен от ученика
    4. Ако съдържанието на PDF файла не е пряко свързано с въпроса, отговорете с: „Съжалявам, прикаченият PDF файл не съдържа отговор на този въпрос.“
    5. Ако са приложени както файл с изображение, така и PDF файл, отговорете, като се позовавате приоритетно на съдържанието на PDF файла
    6. Ако има предишни въпроси и отговори, свързани с въпроса, отговорете, като цитирате съдържанието им и посочите основанията
    7. Ако има съдържание в учебника, свързано с въпроса, отговорете, като цитирате съдържанието на учебника и го посочите като основание
    8. Използвайте подходяща терминология, която ученик от първи курс на гимназията може да разбере
    9. Всички мисли и отговори трябва да се извеждат единствено на български език.
</output_instructions>""")

    # PDF가 없는 경우
    else:
        return SystemMessage(content=f"""{metadata_xml}
<task>Предоставете точен и ясен отговор в рамките на 200 думи, когато отговаряте на въпроси, свързани с науката.</task>
<format_instructions>
    1. Извеждайте в Markdown формат
    2. При писане на формули стриктно спазвайте нотацията на LaTeX
        <good_example>
            (1) $$ \\alpha = \\frac{{\\text{{числител}}}}{{\\text{{знаменател}}}} \\times 100% $$ 
            (2) $$ \\text{{Ag}}^+ $$
        </good_example>
    3. Представяйте URL адресите като обикновен текст (plain text)
</format_instructions>
<output_instructions>
    1. Разделете отговора на две части: „Мисли“ и „Отговор“
    2. В раздела „Мисли“ напишете следното подред:
        - Изисквания: Обобщение на ключовите точки, изисквани във въпроса
        - Основания и цитати: Подреждане на основания и потенциални цитати, които могат да бъдат използвани в отговора
        - Оценка: Оценка на уместността на всяко основание и определяне на посоката на окончателния отговор
    3. Ако има предишни въпроси и отговори, свързани с въпроса, отговорете с приоритетно позоваване на тях, като ги цитирате и посочите основанията
    4. Ако има съдържание в учебника, свързано с въпроса, отговорете, като цитирате съдържанието на учебника и го посочите като основание
    5. Ако няма подходяща информация, отговорете, като използвате собствените знания на LLM
    6. Ако е приложен файл с изображение, анализирайте съдържанието му и отговорете въз основа на него
    7. Ако има извлечен текст от PDF файл, качен от ученика, отговорете на въпроса въз основа на това съдържание
    8. Ако не можете да отговорите, напишете: „Съжалявам. Това е информация, която не ми е известна.“
    9. Използвайте подходяща терминология, която ученик от първи курс на гимназията може да разбере
    10. Всички мисли и отговори трябва да се извеждат единствено на български език.
</output_instructions>""")

# 임베딩 모델 초기화
embeddings_model = HuggingFaceEmbeddings(
    model_name='usmiva/bert-web-bg',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# Vector stores 로드
vectorstore_QNA = FAISS.load_local(os.path.join(BASE_DIR, "faiss_QNA"),
                                   embeddings_model,
                                   allow_dangerous_deserialization=True,
                                   )
vectorstore_QNA.embedding_function = embeddings_model.embed_query

vectorstore_PDF = FAISS.load_local(os.path.join(BASE_DIR, "faiss_PDF"),
                                   embeddings_model,
                                   allow_dangerous_deserialization=True,
                                   )
vectorstore_PDF.embedding_function = embeddings_model.embed_query


@cl.on_chat_start
async def start():
    welcome_messages = [
        ["Здравейте!", "Аз съм", "генеративен", "AI", "чатбот,", "който", "отговаря", "на", "вашите", "въпроси,", "базирайки се", "на", "въпроси", "от по-големи", "ученици", "и", "съдържание", "на", "научни", "материали.\n\n",
         "Попитайте", "за всичко,", "което", "ви интересува,", "докато", "учите", "наука.", "Мога", "да", "отговарям", "на", "въпроси", "за", "средното", "и", "гимназиалното", "образование,", "както", "и", "въпроси,", "свързани", "с", "училищния", "живот.\n\n",
         "Мога", "да", "разпознавам", "картинки", "или", "снимки,", "а", "ако", "качите", "PDF", "файл,", "мога", "да", "анализирам", "и", "неговото", "съдържание.",
         "Помня", "до", "30", "от", "последните", "ни", "разговори.", "Ако", "имате", "допълнителни", "въпроси,", "моля,", "задайте", "ги.🎬\n\n",
         "Поради", "естеството", "на", "AI", "езиковите", "модели,", "отговорите,", "които", "пиша,", "може", "да", "не", "са", "напълно", "точни,", "затова", "ви", "препоръчвам", "да", "потърсите", "съответната", "информация", "в", "учебниците", "и", "справочниците.😊"]
    ]

    for msg in welcome_messages:
        collected_msg = cl.Message(content="", author="science_chatbot")
        for token in msg:
            time.sleep(0.015)
            await collected_msg.stream_token(token + " ")
        await collected_msg.send()

    res = await cl.AskActionMessage(
        content="Моля, кажете ми Вашия прякор.",
        actions=[
            cl.Action(name="continue", payload={"value": "continue"}, label="✅ Кажи ми"),
            cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Пропусни"),
        ],
        author="science_chatbot"
    ).send()

    if res and res.get("payload").get("value") == "continue":
        res_2 = await cl.AskUserMessage(content="Моля, кажете ми Вашия прякор.", timeout=30, author="science_chatbot").send()
        if res_2:
            user_name = res_2['output'].replace("'", "")
            cl.user_session.set("user_name", user_name)
            await cl.Message(content=f"Здравейте, {user_name}! Какво Ви интересува?", author="science_chatbot").send()
    if res and res.get("payload").get("value") == "cancel":
        cl.user_session.set("user_name", "Нерегистриран")
        await cl.Message(content="Здравейте! Какво Ви интересува?", author="science_chatbot").send()

    retriever_QNA = vectorstore_QNA.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 4, 'fetch_k': 40, 'lambda_mult': 0.95}
    )

    retriever_PDF = vectorstore_PDF.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 6, 'fetch_k': 60, 'lambda_mult': 0.95}
    )

    history_dict = {}

    def get_session_history(session_id: str):
        if session_id not in history_dict:
            history_dict[session_id] = InMemoryChatMessageHistory()
        return history_dict[session_id]

    cl.user_session.set("session_id", cl.user_session.get("id"))
    cl.user_session.set("get_session_history", get_session_history)
    cl.user_session.set("retriever_QNA", retriever_QNA)
    cl.user_session.set("retriever_PDF", retriever_PDF)
    cl.user_session.set("messages", [])
    cl.user_session.set("uploaded_pdf_docs", [])


@cl.on_message
async def main(message: cl.Message):
    messages = cl.user_session.get("messages")
    uploaded_pdf_docs = cl.user_session.get("uploaded_pdf_docs", [])
    user_name = cl.user_session.get("user_name", "Нерегистриран")

    def load_memory(input_dict: Dict[str, Any]) -> List[BaseMessage]:
        return trim_messages(messages, token_counter=len, max_tokens=60,
                             strategy="last", start_on="human", include_system=True, allow_partial=False)

    images = []
    pdfs = []

    for file in message.elements:
        if hasattr(file, 'mime') and hasattr(file, 'path') and hasattr(file, 'name'):
            file_path_lower = file.path.lower()
            file_name_lower = file.name.lower()

            # 이미지 파일 판단: MIME 타입 또는 원본 파일명 확장자 기반
            is_image = (
                    "image" in file.mime or
                    file_name_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.heic', '.heif', '.bmp', '.webp'))
            )

            # PDF 파일 판단
            is_pdf = "pdf" in file.mime or file_name_lower.endswith('.pdf')

            if is_image and not is_pdf:
                images.append(file)
            elif is_pdf:
                pdfs.append(file)

    image_base64 = None
    image_path_in_db = None
    pdf_path_in_db = None
    pdf_documents = []
    use_pdf_only = False

    # PDF 처리
    if pdfs:
        pdf_path = pdfs[0].path
        saved_path, error = save_pdf_file(pdf_path, user_name)

        if error:
            error_msg = cl.Message(content=f"❌ Възникна грешка при запазване на PDF файла.:\n{error}", author="science_chatbot")
            await error_msg.send()
        else:
            pdf_path_in_db = saved_path
            pdf_filename = os.path.basename(pdf_path_in_db)

            pdf_documents, error = extract_text_from_pdf_with_opendataloader(pdf_path, pdf_filename)

            if error:
                error_msg = cl.Message(content=f"❌ Възникна грешка при обработката на PDF файла.:\n{error}", author="science_chatbot")
                await error_msg.send()
            else:
                use_pdf_only = True
                uploaded_pdf_docs = pdf_documents
                cl.user_session.set("uploaded_pdf_docs", uploaded_pdf_docs)

                pdf_info_msg = cl.Message(
                    content=f"✅ PDF файлът беше успешно качен. ({len(pdf_documents)} извлечени фрагмента).\n Ще отговоря на въпроса въз основа на съдържанието на този PDF.",
                    author="science_chatbot"
                )
                await pdf_info_msg.send()

    # 이미지 처리
    if images:
        image_path = images[0].path

        # 1. 이미지 파일 영구 저장
        saved_path, error = save_image_file(image_path, user_name)

        if error:
            error_msg = cl.Message(content=f"❌ Възникна грешка при запазване на изображението.:\n{error}", author="science_chatbot")
            await error_msg.send()
        else:
            image_path_in_db = saved_path

            # 2. 이미지 변환 (멀티모달 LLM용 - HEIC도 JPEG로 변환)
            try:
                jpeg_base64, error = convert_image_to_jpeg(image_path)

                if error:
                    error_msg = cl.Message(content=f"❌ Възникна грешка при конвертиране на изображение.:\n{error}", author="science_chatbot")
                    await error_msg.send()
                    image_base64 = None
                else:
                    image_base64 = jpeg_base64

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                error_msg = cl.Message(content=f"❌ Възникна изключение при обработката на изображението.: {str(e)}\n{error_trace}",
                                       author="science_chatbot")
                await error_msg.send()
                image_base64 = None

    # 문서 검색
    if use_pdf_only == True:
        docs_QNA = []
        docs_PDF = []
    else:
        retriever_QNA = cl.user_session.get("retriever_QNA")
        retriever_PDF = cl.user_session.get("retriever_PDF")
        docs_QNA = retriever_QNA.invoke(message.content)
        docs_PDF = retriever_PDF.invoke(message.content)

    # 컨텍스트 메시지 생성
    if use_pdf_only == True:
        context_messages = create_context_messages(uploaded_pdf_docs=uploaded_pdf_docs)
    else:
        context_messages = create_context_messages(docs_QNA, docs_PDF, uploaded_pdf_docs)

    # LangChain 0.3 표준: 이미지 포함 HumanMessage 생성
    human_message_content = []

    # 이미지가 있으면 먼저 추가
    if image_base64:
        human_message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })

    # 텍스트 추가
    human_message_content.append({
        "type": "text",
        "text": message.content
    })

    current_human_message = HumanMessage(content=human_message_content)

    # 메모리에 저장
    messages.append(current_human_message)

    # 시스템 프롬프트 선택 - 동적 생성
    system_prompt = create_dynamic_system_prompt(
        user_name=user_name,
        use_pdf_only=use_pdf_only,
        uploaded_pdf_docs=uploaded_pdf_docs
    )

    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        *context_messages,
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="input"),
    ])

    base_chain = prompt | llm | StrOutputParser()

    get_session_history = cl.user_session.get("get_session_history")
    session_id = cl.user_session.get("session_id")

    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # 응답 생성 및 스트리밍
    # 1. 일단 빈 메시지를 전송해 둡니다 (스트리밍 준비)
    msg = cl.Message(content="", author="science_chatbot")
    await msg.send()
    
    collected_output = ""
    start_time = time.time()
    
    # 첫 번째 토큰 수신 여부 플래그
    first_token_received = False
    prefix_buffer = ""      # [추가] Qwen 마크다운 필터링 버퍼
    prefix_handled = False  # [추가] Qwen 마크다운 필터링 완료 플래그
    
    # 3초 뒤에 실행될 비동기 함수 정의
    async def show_delayed_waiting_message():
        await asyncio.sleep(3)  # 3초 대기
        # 3초 뒤에도 아직 첫 토큰이 안 왔다면 안내 문구 출력
        if not first_token_received:
            msg.content = "Може да отнеме около 30 секунди, докато се генерира отговорът, тъй като анализирам предишни въпроси и отговори, както и съдържанието на съответните учебници. Ако прикачите изображение, анализът може да отнеме повече от минута, затова, моля, изчакайте..."
            await msg.update()
    
    # 비동기 타이머 시작 (백그라운 실행)
    waiting_task = asyncio.create_task(show_delayed_waiting_message())
    
    try:
        user_input_for_chain = [current_human_message]
    
        async for chunk in chain_with_history.astream(
                {"input": user_input_for_chain},
                config={"configurable": {"session_id": session_id}}
        ):
            # 1. 콘텐츠 추출
            if hasattr(chunk, "content"):
                raw_content = chunk.content
            else:
                raw_content = chunk
    
            # 2. 텍스트 변환
            text_content = ""
            if isinstance(raw_content, list):
                text_content = ''.join([
                    item.get('text', '') if isinstance(item, dict) else str(item)
                    for item in raw_content
                ])
            elif isinstance(raw_content, str):
                text_content = raw_content
            else:
                text_content = str(raw_content)
    
            # 3. 출력 및 저장
            if text_content:
                # [기존 로직 유지] 첫 토큰이 도착했을 때 처리
                if not first_token_received:
                    first_token_received = True  # 플래그 변경
                    waiting_task.cancel()  # 3초 타이머 취소
    
                    # 만약 이미 안내 문구가 화면에 떠 있다면 지워줌
                    if msg.content and "Може да отнеме" in msg.content:
                        msg.content = ""
                        await msg.update()
    
                # [수정된 부분] Qwen 마크다운 필터링 후 스트리밍
                if not prefix_handled:
                    prefix_buffer += text_content
                    stripped_buffer = prefix_buffer.lstrip().lower()
    
                    # '```markdown' 또는 '```' 기호가 들어오는지 확인하며 대기
                    if "```markdown".startswith(stripped_buffer) or "```".startswith(stripped_buffer):
                        continue
                    
                    # 마크다운 기호가 아니거나 넘어선 텍스트면 스트리밍 시작
                    prefix_handled = True
                    
                    if stripped_buffer.startswith("```markdown"):
                        text_to_stream = prefix_buffer.lstrip()[11:].lstrip()
                    elif stripped_buffer.startswith("```"):
                        text_to_stream = prefix_buffer.lstrip()[3:].lstrip()
                    else:
                        text_to_stream = prefix_buffer
    
                    if text_to_stream:
                        collected_output += text_to_stream
                        await msg.stream_token(text_to_stream)
                else:
                    # 앞부분 필터링이 끝난 이후에는 정상적으로 스트리밍
                    collected_output += text_content
                    await msg.stream_token(text_content)
    
            elapsed_time = time.time() - start_time
            if elapsed_time > 180:
                raise asyncio.TimeoutError("Генерирането се прекъсва, тъй като времето за създаване надвиши 3 минути.")
    
        # 응답 완료 후, 맨 끝에 남아있는 닫는 백틱(```) 제거
        final_output = collected_output.strip()
        if final_output.endswith("```"):
            final_output = final_output[:-3].strip()
    
        messages.append(AIMessage(content=final_output))
        cl.user_session.set("messages", messages)
        msg.content = final_output # 명시적으로 최종 텍스트 덮어쓰기
    
    except asyncio.TimeoutError as e:
        if collected_output:
            messages.append(AIMessage(content=collected_output))
            cl.user_session.set("messages", messages)
        await msg.stream_token("\n\n[시간 초과로 응답이 중단되었습니다]")
    
    except Exception as e:
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        # 에러 메시지 문자열 생성
        error_text = f"\n\n[오류가 발생했습니다: {str(e)}]"
        # 에러 발생 시 안내 문구가 남아있다면 지움
        if msg.content and "Може да отнеме" in msg.content:
            msg.content = ""
            await msg.update()
        # 화면에 에러 메시지 출력
        await msg.stream_token(error_text)
        #  DB 저장을 위해 변수에 에러 내용 추가
        collected_output += error_text
    
    finally:
        # 혹시라도 루프가 비정상 종료되거나 끝났을 때 태스크가 살아있다면 취소
        if not waiting_task.done():
            waiting_task.cancel()
    
    await msg.update()

    session_id = cl.user_session.get("id")
    accuracy = 0
    satisfaction = 0

    connection = pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        db=os.getenv('DB_NAME'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    # try:
    #     with connection.cursor() as cursor:
    #         sql = """INSERT INTO datalog_gen_3
    #                  (user_name, session_id, student_question, answer, accuracy, satisfaction, image_path, pdf_path, \
    #                   selected_similarity)
    #                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""

    #         cursor.execute(sql, (
    #             user_name,
    #             session_id,
    #             message.content,
    #             collected_output,
    #             accuracy,
    #             satisfaction,
    #             image_path_in_db if image_path_in_db else None,
    #             pdf_path_in_db if pdf_path_in_db else None,
    #             None
    #         ))
    #     connection.commit()
    # finally:
    #     connection.close()

    # 평가 버튼 표시
    actions = [
        cl.Action(name="correct_btn", payload={"value": "correct"}, label="✅ Точно обяснение"),
        cl.Action(name="wrong_btn", payload={"value": "wrong"}, label="❌ Грешно обяснение"),
        cl.Action(name="satisfied_btn", payload={"value": "satisfied"}, label="👍 Обяснение с подходящо ниво"),
        cl.Action(name="dissatisfied_btn", payload={"value": "dissatisfied"}, label="👎 Твърде лесно или твърде трудно обяснение")
    ]
    await cl.Message(content="„Моля, оценете точността на отговора и адекватността на нивото на обяснение.", actions=actions, author="science_chatbot").send()

    # 유사 질문-답변 쌍을 Action으로 제시
    if not use_pdf_only:
        # 1. 높은 유사도로 필터링된 질문 (0.5 이상, 최대 2개)
        docs_high_similarity = vectorstore_QNA.similarity_search_with_relevance_scores(
            message.content,
            k=10  # 충분히 많이 가져온 후 필터링
        )

        high_similarity_actions = []
        for doc, score in docs_high_similarity:
            if score >= 0.5 and len(high_similarity_actions) < 2:  # 0.5 이상, 최대 2개
                if doc.metadata.get("답변"):
                    question = doc.page_content
                    answer = doc.metadata.get("답변")
                    payload = {
                        "question": question,
                        "answer": answer,
                        "similarity": float(score)
                    }
                    high_similarity_actions.append(
                        cl.Action(
                            name="similar_question",
                            payload=payload,
                            label=f"❓ [유사도: {score:.2f}] {question[:80]}"
                        )
                    )

        # 2. MMR 검색 (항시 4개, 기존 설정 유지)
        docs_QNA = vectorstore_QNA.max_marginal_relevance_search(
            message.content,
            k=4,
            fetch_k=40,
            lambda_mult=0.95  # 기존 최적값 유지
        )

        mmr_actions = []
        for doc in docs_QNA:
            if doc.metadata.get("답변"):
                question = doc.page_content
                answer = doc.metadata.get("답변")
                payload = {
                    "question": question,
                    "answer": answer,
                    "similarity": 0  # MMR은 점수 없음
                }
                mmr_actions.append(
                    cl.Action(
                        name="similar_question",
                        payload=payload,
                        label=f"❓ {question[:80]}"
                    )
                )

        # 3. 메시지 표시
        if high_similarity_actions:
            await cl.Message(
                content="💡 Директно свързани въпроси:",
                actions=high_similarity_actions,
                author="science_chatbot"
            ).send()

        if mmr_actions:
            await cl.Message(
                content="🔍 Въпроси от различни гледни точки:",
                actions=mmr_actions,
                author="science_chatbot"
            ).send()

@cl.action_callback("similar_question")
async def on_similar_question(action: cl.Action):
    """
    유사 질문 Action을 클릭했을 때 답변을 보여주고 DB에 로그를 남깁니다.
    """
    payload = action.payload
    question = payload.get("question")
    answer = payload.get("answer")
    similarity = payload.get("similarity")

    await cl.Message(content=f"**Избран въпрос:**\n{question}\n\n**Отговор:**\n{answer}", author="science_chatbot").send()

    user_name = cl.user_session.get("user_name", "Нерегистриран")
    session_id = cl.user_session.get("id")

    connection = None
    # try:
    #     connection = pymysql.connect(
    #         host=os.getenv('DB_HOST'),
    #         user=os.getenv('DB_USER'),
    #         password=os.getenv('DB_PASSWORD'),
    #         db=os.getenv('DB_NAME'),
    #         charset='utf8mb4',
    #         cursorclass=pymysql.cursors.DictCursor
    #     )
    #     with connection.cursor() as cursor:
    #         sql = """INSERT INTO datalog_gen_3
    #                  (user_name, session_id, student_question, answer, accuracy, satisfaction, selected_similarity)
    #                  VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    #         cursor.execute(sql, (user_name, session_id, question, answer, 0, 0, float(similarity)))
    #     connection.commit()
    # except Exception as e:
    #     print(f"DB Error on similar question logging: {e}")
    #     await cl.Message(content=f"DB 관련 오류가 발생했습니다: {e}", author="science_chatbot").send()
    # finally:
    #     if connection:
    #         connection.close()


@cl.action_callback("correct_btn")
async def on_correct(action):
    accuracy = 2
    session_id = cl.user_session.get("id")
    connection = None
    # try:
    #     connection = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'),
    #                                  password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'),
    #                                  charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    #     with connection.cursor() as cursor:
    #         cursor.execute("SELECT MAX(input_time) as max_time FROM datalog_gen_3 WHERE session_id = %s", (session_id,))
    #         result = cursor.fetchone()
    #         if result and result['max_time']:
    #             max_time = result['max_time']
    #             sql = "UPDATE datalog_gen_3 SET accuracy = %s WHERE session_id = %s AND input_time = %s"
    #             cursor.execute(sql, (accuracy, session_id, max_time))
    #     connection.commit()
    # finally:
    #     if connection:
    #         connection.close()
    await cl.Message(content="Регистрирах, че отговорът е точен.", author="science_chatbot").send()


@cl.action_callback("wrong_btn")
async def on_wrong(action):
    accuracy = 1
    session_id = cl.user_session.get("id")
    connection = None
    # try:
    #     connection = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'),
    #                                  password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'),
    #                                  charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    #     with connection.cursor() as cursor:
    #         cursor.execute("SELECT MAX(input_time) as max_time FROM datalog_gen_3 WHERE session_id = %s", (session_id,))
    #         result = cursor.fetchone()
    #         if result and result['max_time']:
    #             max_time = result['max_time']
    #             sql = "UPDATE datalog_gen_3 SET accuracy = %s WHERE session_id = %s AND input_time = %s"
    #             cursor.execute(sql, (accuracy, session_id, max_time))
    #     connection.commit()
    # finally:
    #     if connection:
    #         connection.close()
    await cl.Message(content="Регистрирах, че отговорът е грешен.", author="science_chatbot").send()


@cl.action_callback("satisfied_btn")
async def on_accurate(action):
    satisfaction = 2
    session_id = cl.user_session.get("id")
    connection = None
    # try:
    #     connection = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'),
    #                                  password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'),
    #                                  charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    #     with connection.cursor() as cursor:
    #         cursor.execute("SELECT MAX(input_time) as max_time FROM datalog_gen_3 WHERE session_id = %s", (session_id,))
    #         result = cursor.fetchone()
    #         if result and result['max_time']:
    #             max_time = result['max_time']
    #             sql = "UPDATE datalog_gen_3 SET satisfaction = %s WHERE session_id = %s AND input_time = %s"
    #             cursor.execute(sql, (satisfaction, session_id, max_time))
    #     connection.commit()
    # finally:
    #     if connection:
    #         connection.close()
    await cl.Message(content="Регистрирах, че нивото на обяснение е подходящо.", author="science_chatbot").send()


@cl.action_callback("dissatisfied_btn")
async def on_not_accurate(action):
    satisfaction = 1
    session_id = cl.user_session.get("id")
    connection = None
    # try:
    #     connection = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'),
    #                                  password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'),
    #                                  charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    #     with connection.cursor() as cursor:
    #         cursor.execute("SELECT MAX(input_time) as max_time FROM datalog_gen_3 WHERE session_id = %s", (session_id,))
    #         result = cursor.fetchone()
    #         if result and result['max_time']:
    #             max_time = result['max_time']
    #             sql = "UPDATE datalog_gen_3 SET satisfaction = %s WHERE session_id = %s AND input_time = %s"
    #             cursor.execute(sql, (satisfaction, session_id, max_time))
    #     connection.commit()
    # finally:
    #     if connection:
    #         connection.close()
    await cl.Message(content="Регистрирах, че нивото на обяснение е твърде лесно или твърде трудно.", author="science_chatbot").send()


@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    # MCP 연결 시 로그 출력
    print(f"MCP connection established: {connection}")


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    # MCP 연결 종료 시 로그 출력
    print(f"MCP connection terminated: {name}")


if __name__ == "__main__":
    cl.run()
