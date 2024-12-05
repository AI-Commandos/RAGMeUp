import sqlite3
import os

# create own sqlite database
def create_database(db_path):
    """
    Create a SQLite database and tables for storing files.
    """
    # Connecting to a SQLite Database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a table for storing files
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,  
        file_name TEXT NOT NULL,               
        file_data BLOB NOT NULL                
    )
    """)

    print(f"Database and table created successfully at {db_path}")

  
    conn.commit()
    conn.close()

# Insert files into the database
def insert_file_into_database(db_path, file_path):
    """
    :param db_path: Database File Path
    :param file_path: Example File Path
    """
    # Connecting to a SQLite Database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(file_path, 'rb') as f:
        file_data = f.read()


    cursor.execute("""
    INSERT INTO files (file_name, file_data)
    VALUES (?, ?)
    """, (os.path.basename(file_path), file_data))

    print(f"Inserted {os.path.basename(file_path)} into database.")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    database_path = '/Users/wanghanyue/desktop/my_pdf_database.db'
    create_database(database_path)

    example_files = ["/Users/wanghanyue/desktop/dsbe/pm/machine_learning/ppt/8.Ensemble_Techniques.pdf",
                 "/Users/wanghanyue/desktop/dsbe/pm/machine_learning/ppt/4.Regression.pdf",
                "/Users/wanghanyue/desktop/dsbe/pm/machine_learning/ppt/3.Classification.pdf",
                  "/Users/wanghanyue/desktop/dsbe/pm/machine_learning/ppt/2.Feature_Engineering.pdf"]  # Replace with your actual PDF file path

    # insert file
    for example_file in example_files:
        if os.path.exists(file_path):  
            insert_file_into_database(database_path, file_path)
        else:
            print(f"File {file_path} does not exist. Skipping.")
