-- conf/evolutions/ragmeup/1.sql

# --- !Ups
CREATE TABLE IF NOT EXISTS chat_logs (
    id TEXT,
    message_offset INTEGER,
    text TEXT,
    role TEXT,
    reply TEXT,
    documents TEXT,
    rewritten INTEGER,
    question TEXT,
    fetched_new_documents INTEGER,
    PRIMARY KEY (id, message_offset)
);

CREATE TABLE IF NOT EXISTS feedback (
    chat_id TEXT,
    message_offset INTEGER,
    feedback INTEGER,
    FOREIGN KEY (chat_id, message_offset) REFERENCES chat_logs(id, message_offset)
);

# --- !Downs
DROP TABLE IF EXISTS chat_logs;
DROP TABLE IF EXISTS feedback;
