import streamlit as st
import os
import base64
import glob
import time
import uuid
import io
from PIL import Image
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
NUMBER = 5

try:
    api_key = st.secrets["ZHIPU_API_KEY"]
except FileNotFoundError:
    api_key = os.environ.get("ZHIPU_API_KEY", "")
if not api_key:
    st.error("请配置 API Key！可以在 .streamlit/secrets.toml 中配置。")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)
st.set_page_config(page_title="Offer选大米助手 Pro", layout="wide")

client = OpenAI(
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)
MODEL_NAME = "glm-4.6v-flashx"  


IMAGE_FOLDER = "./images"
DB_PATH = "./chroma_db"




@st.cache_resource
def load_embed_model():

    print("正在加载本地 Embedding 模型 (BGE-M3)...")
    return SentenceTransformer('BAAI/bge-m3')


embed_model = load_embed_model()

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="offer_rag_collection")


def encode_image(image_path, max_size=1024):

    try:
        with Image.open(image_path) as img:

            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            width, height = img.size
            if max(width, height) > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"图片压缩失败: {e}，尝试使用原图...")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def get_embedding(text):
    return embed_model.encode(text).tolist()


def analyze_image_content(image_path):
    base64_img = encode_image(image_path)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": """
                         你是一个数据提取器。这张图包含用户的Offer信息，但也包含无关的个人背景（如“本人某地人”、“纠结中”）。
                         你的任务是只提取 Offer 里的硬核数据。

                         请严格按照以下 JSON 格式输出（不要输出 Markdown 代码块，只输出纯 JSON 字符串）：
                         {
                            "source_file": "文件名",
                            "offers": [
                                {
                                    "company": "公司名称",
                                    "department": "部门/岗位",
                                    "location": "工作地点(Base)",
                                    "salary": "薪资公式(如(n+3)*16，原样摘录)",
                                    "welfare": "签字费/房补/公积金等",
                                    "pros_cons": "原文提到的优缺点"
                                }
                            ]
                         }
                         """},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }
            ],
            top_p=0.1,  # 极低随机性，保证格式稳定
            temperature=0.1
        )

        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return content
    except Exception as e:
        print(f"解析图片失败 {image_path}: {e}")
        return ""


def build_database():
    """建库流程（支持增量更新）"""
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        st.error(f"请先建立 {IMAGE_FOLDER} 文件夹并放入图片！")
        return

    image_files = glob.glob(os.path.join(IMAGE_FOLDER, "*.*"))

    # 1. 增量检查
    existing_data = collection.get()
    existing_paths = set()
    if existing_data and existing_data['metadatas']:
        for meta in existing_data['metadatas']:
            existing_paths.add(meta['image_path'])

    files_to_add = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f not in existing_paths]

    if not files_to_add:
        st.info("数据库已是最新，无需更新。")
        return

    # 2. 开始处理
    progress_bar = st.progress(0)
    status_text = st.empty()

    ids = []
    embeddings = []
    metadatas = []
    new_count = 0

    for i, img_path in enumerate(files_to_add):
        status_text.text(f"正在清洗并入库: {os.path.basename(img_path)} ...")

        # 获取清洗后的 JSON 文本
        caption = analyze_image_content(img_path)
        if not caption: continue

        ids.append(str(uuid.uuid4()))
        embeddings.append(get_embedding(caption))
        metadatas.append({"image_path": img_path, "caption": caption})

        new_count += 1
        progress_bar.progress((i + 1) / len(files_to_add))

    if ids:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        st.success(f" 更新完成！清洗并存入了 {new_count} 张 Offer 数据。")
        time.sleep(1)

    status_text.empty()
    progress_bar.empty()


def chat_pipeline(user_query):
    """
    【RAG 精准狙击版】：白名单过滤 + 强制Markdown格式修复
    只回答用户点名的公司，剔除无关干扰项。
    """
    print(f"\n正在思考问题: {user_query} ...")

    query_vec = get_embedding(user_query)
    results = collection.query(query_embeddings=[query_vec], n_results=NUMBER)

    if not results['metadatas'] or not results['metadatas'][0]:
        return "数据库里好像没有相关的 Offer 图片。", []

    content = []
    retrieved_items = results['metadatas'][0]
    display_images = []
    MAX_IMAGES_TO_LLM = 5

    context_str_list = []

    # === 1. 构建白名单 (White-List) ===
    # 定义我们关注的所有大厂关键词
    known_companies = ["快手", "字节", "腾讯", "阿里", "蚂蚁", "美团", "拼多多", "华为", "百度", "小红书", "京东",
                       "网易","讯飞","滴滴"]

    # 找出用户问句里提到的公司
    target_companies = [c for c in known_companies if c in user_query]

    # 生成过滤指令
    if target_companies:
        # 如果用户明确提到了公司名（如“字节、快手”），开启严格过滤模式
        filter_instruction = f"""
        【严格过滤指令】
        用户只对以下公司感兴趣：【{'、'.join(target_companies)}】。
        请执行“白名单过滤”：
        1. 仔细阅读下方数据源。
        2. 如果数据源里的 Offer 属于上述名单，请提取。
        3. 如果数据源是“阿里”、“拼多多”、“蚂蚁”等**不在名单内**的公司，直接忽略，**严禁**写入表格！
        4. 表格里只能出现：{'、'.join(target_companies)}。
        """
    else:
        # 如果用户没提具体公司（如“帮我对比一下库里的Offer”），则不过滤
        filter_instruction = "用户未指定具体公司，请列出所有检索到的高质量 Offer。"

    # === 2. 组装数据 ===
    for i, item in enumerate(retrieved_items):
        try:
            img_path = item["image_path"]
            json_data = item["caption"]

            display_images.append(img_path)

            # 调试显示
            #st.write(f"####  数据源 {i + 1}：")
            #with st.expander("查看原始 JSON"):
            #    st.code(json_data, language='json')

            if i < MAX_IMAGES_TO_LLM:
                base64_img = encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                })
                context_str_list.append(f"== 数据源 {i + 1} (含图) ==\n{json_data}")
            else:
                context_str_list.append(f"== 数据源 {i + 1} (仅文本) ==\n{json_data}")

        except Exception as e:
            print(f"处理检索结果 {i} 出错: {e}")

    full_context = "\n\n".join(context_str_list)

    # === 3. 最终 Prompt ===
    prompt_text = f"""
    用户问题："{user_query}"

    {filter_instruction}

    我提供了 {len(retrieved_items)} 份 Offer 数据。请严格按照要求输出。

    **要求一：输出表格 (Markdown)**
    必须输出标准的 Markdown 表格，格式如下（不要合并单元格，每一行对应一个Offer）：
    | 公司 | 部门/岗位 | 地点 | 薪资公式 | 总包估算 | 优缺点 |
    |---|---|---|---|---|---|
    | ... | ... | ... | ... | ... | ... |

    **要求二：内容清洗**
    - 薪资公式：原样摘录 JSON 里的公式（如 `(n+3)*16`）。
    - 总包估算：如果 n 未知，保留 n 的公式形式。
    - 优缺点：精简提取，不要长篇大论。

    **要求三：最终建议**
    针对用户点名的这几家公司，给出一句简短的选择建议。

    === 待处理数据源 ===
    {full_context}
    """

    content.append({"type": "text", "text": prompt_text})

    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个只听指令的数据分析师，严格执行过滤要求。"},
            {"role": "user", "content": content}
        ],
        stream=False,
        top_p=0.2,
        temperature=0.1
    )

    return final_response.choices[0].message.content, display_images

def main():
    st.title("AI Offer 选大米助手 Pro")

    with st.sidebar:
        st.header("控制台")
        if st.button("读取新图片并更新库"):
            with st.spinner("正在压缩图片并清洗数据，请稍候..."):
                build_database()

        st.info(f"每次检索 {NUMBER} 张最相关的 Offer")
        st.caption("提示：初次运行请先点击更新库")

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 处理输入
    if prompt := st.chat_input("请输入问题 (例如: 对比一下字节和快手，算一下总包)"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.status("AI 正在检索并疯狂计算中...", expanded=True) as status:
                try:
                    answer, ref_images = chat_pipeline(prompt)

                    if ref_images:
                        st.write(f"检索到 {len(ref_images)} 张相关 Offer：")
                        cols = st.columns(len(ref_images))
                        for idx, img_path in enumerate(ref_images):
                            # 使用 use_container_width=True 修复布局报错
                            cols[idx].image(img_path, caption=f"Offer {idx + 1}", width='content')

                    status.update(label="分析完成！", state="complete", expanded=False)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"发生错误: {e}")


if __name__ == "__main__":
    main()