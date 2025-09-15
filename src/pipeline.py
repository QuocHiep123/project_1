# src/pipeline.py
from langchain.chains.qa_with_sources.stuff_prompt import template
from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from sympy.physics.units import temperature
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

def create_llm():
    '''
    Khởi tạo ngôn ngữ lớn từ LLM của hugging face
    '''
    model_name = 'google/flan-t5-small'
    print('đang tải model LLMs')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # tạo pipeline
    pipe = pipeline(
        'text2text-generation',
        model = model,
        tokenizer= tokenizer,
        max_length = 512
    )
    llm = HuggingFacePipeline(pipeline = pipe)
    return llm
def create_qa_chain(vector_store, llm):
    '''
    tạo ra chuỗi chain để hỏi đáp hoàn chỉnh
    '''
    retriever = vector_store.as_retriever(
        search_kwargs = {'k': 2}
    )

    # tạo prompt để hướng dẫn llm trả lời
    prompt_template = '''
    Bạn là một trợ lý AI hữu ích. Chỉ sử dụng thông tin được cung cấp trong phần ngữ cảnh dưới đây để trả lời câu hỏi.
    Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói rằng "Tôi không tìm thấy thông tin để trả lời câu hỏi này trong tài liệu.".
    Không được tự bịa ra câu trả lời.

    Ngữ cảnh:
    {context}

    Câu hỏi:
    {question}

    Câu trả lời hữu ích:
    '''
    PROMPT = PromptTemplate(
        template = prompt_template,
        input_varible = ['context', 'question']
    )
    # tạo chuỗi Retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= 'stuff',
        retriever = retriever,
        chain_type_kwargs= {'prompt' : PROMPT},
        return_source_documents = True,
    )
    return qa_chain
def answer_question(qa_chain, question: str) -> dict:
    '''
    sử dụng chain hỏi đáp để trả lời câu hỏi cụ thể
    '''
    result = qa_chain.invoke({"query" : question})
    return result