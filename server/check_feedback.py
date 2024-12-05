import sqlite3

# Path to your feedback database
db_path = 'feedback.db'

def check_feedback_data():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check the data in the Feedback table
        cursor.execute("SELECT * FROM Feedback")
        rows = cursor.fetchall()

        if rows:
            print("Feedback Data:")
            for row in rows:
                print(row)
        else:
            print("No feedback data found in the database.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_feedback_data()
