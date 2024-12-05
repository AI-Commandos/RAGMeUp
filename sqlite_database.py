import sqlite3
import os

# 创建 SQLite 数据库和表
def create_database(db_path):
    """
    创建 SQLite 数据库和用于存储 PDF 文件的表。
    """
    # 连接到 SQLite 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建一个表用于存储 PDF 文件
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pdf_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自动递增的唯一标识
        file_name TEXT NOT NULL,               -- PDF 文件名
        file_data BLOB NOT NULL                -- PDF 文件的二进制数据
    )
    """)

    print(f"Database and table created successfully at {db_path}")

    # 提交更改并关闭连接
    conn.commit()
    conn.close()

# 插入 PDF 文件到数据库
def insert_pdf_into_database(db_path, pdf_path):
    """
    将 PDF 文件插入到数据库中。

    :param db_path: 数据库文件路径
    :param pdf_path: PDF 文件路径
    """
    # 连接到 SQLite 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 读取 PDF 文件为二进制数据
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()

    # 插入 PDF 文件到表中
    cursor.execute("""
    INSERT INTO pdf_files (file_name, file_data)
    VALUES (?, ?)
    """, (os.path.basename(pdf_path), pdf_data))

    print(f"Inserted {os.path.basename(pdf_path)} into database.")

    # 提交更改并关闭连接
    conn.commit()
    conn.close()

# 示例运行
if __name__ == "__main__":
    # 数据库文件路径
    database_path = '/Users/wanghanyue/desktop/my_pdf_database.db'

    # 创建数据库和表
    create_database(database_path)

    # PDF 文件路径
    pdf_files = ["/Users/wanghanyue/desktop/dsbe/pm/machine_learning/ppt/8.Ensemble_Techniques.pdf",
                 "/Users/wanghanyue/desktop/dsbe/pm/machine_learning/ppt/4.Regression.pdf",
                "/Users/wanghanyue/desktop/dsbe/pm/machine_learning/ppt/3.Classification.pdf",
                  "/Users/wanghanyue/desktop/dsbe/pm/machine_learning/ppt/2.Feature_Engineering.pdf"]  # 替换为您实际的 PDF 文件路径

    # 插入 PDF 文件
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):  # 确保 PDF 文件存在
            insert_pdf_into_database(database_path, pdf_file)
        else:
            print(f"File {pdf_file} does not exist. Skipping.")
