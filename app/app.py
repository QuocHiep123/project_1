# app/app.py

import gradio as gr
import time

# Import các hàm xử lý từ thư mục src
from src.data_processor import load_documents, chunk_documents
from src.vector_store_manager import create_vector_store
from src.pipeline import create_llm, create_qa_chain, answer_question

# Khởi tạo LLM ngay khi ứng dụng khởi động để tiết kiệm thời gian
llm = create_llm()


def process_file(file):
    if file is None:
        return None, "Vui lòng tải lên một file."

    file_path = file.name
    print(f"Bắt đầu xử lý file: {file_path}")

    try:
        docs = load_documents(file_path)
        if not docs:
            return None, f"Không thể đọc nội dung từ file {file.name}."

        chunks = chunk_documents(docs)
        if not chunks:
            return None, "Tài liệu quá ngắn để có thể phân đoạn."

        vector_store = create_vector_store(chunks)
        print("Đã xử lý file thành công!")
        return vector_store, f"Đã xử lý thành công file: {file.name}"
    except Exception as e:
        return None, f"Đã xảy ra lỗi khi xử lý file: {e}"


def chat_with_document(question, history, vector_store):
    if vector_store is None:
        history.append((question, "Lỗi: Vui lòng tải lên và nhấn 'Xử lý' một tài liệu trước."))
        return history

    # Tạo chuỗi QA với LLM đã được khởi tạo
    qa_chain = create_qa_chain(vector_store, llm)

    # Lấy câu trả lời
    result = answer_question(qa_chain, question)
    answer = result.get("result", "Không thể tạo câu trả lời.")
    history.append((question, answer))
    return history


with gr.Blocks() as demo:
    vector_store_state = gr.State(None)

    gr.Markdown("# 💬 Hỏi Đáp Tài Liệu Của Bạn (RAG Demo)")
    gr.Markdown("Tải lên file .PDF hoặc .TXT của bạn, nhấn 'Xử lý', sau đó đặt câu hỏi về nội dung bên trong.")

    with gr.Row():
        with gr.Column(scale=3):
            file_uploader = gr.File(label="Tải tài liệu lên", file_types=['.pdf', '.txt'])
            process_button = gr.Button("Xử lý", variant="primary")
            status_display = gr.Textbox(label="Trạng thái xử lý", interactive=False)

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="Cuộc hội thoại", bubble_full_width=False, value = [])
            msg = gr.Textbox(label="Đặt câu hỏi của bạn ở đây")
            clear = gr.ClearButton([msg, chatbot])

    # Định nghĩa luồng tương tác
    process_button.click(
        fn=process_file,
        inputs=[file_uploader],
        outputs=[vector_store_state, status_display],
        show_progress="full"
    )

    msg.submit(
        fn=chat_with_document,
        inputs=[msg, chatbot, vector_store_state],
        outputs=[chatbot],
    ).then(lambda: gr.update(value=""), outputs=[msg])

if __name__ == "__main__":
    demo.launch(debug=True)