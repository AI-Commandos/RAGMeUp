package models

import play.api.Configuration
import play.api.db.slick.{DatabaseConfigProvider, HasDatabaseConfigProvider}
import play.db.NamedDatabase
import slick.jdbc.{JdbcProfile, SQLiteProfile, PostgresProfile}

import javax.inject.{Inject, Singleton}

case class ChatLog(
  id: String, messageOffset: Int, text: String, role: String, reply: String,
  documents: String, rewritten: Boolean, question: String, fetchedNewDocuments: Boolean
)
case class Feedback(
  chatId: String, messageOffset: Int, feedback: Boolean
)

@Singleton
class SQLTables @Inject() (@NamedDatabase("ragmeup") protected val dbConfigProvider: DatabaseConfigProvider, config: Configuration)
  extends HasDatabaseConfigProvider[JdbcProfile] {

  val selectedProfile: JdbcProfile = config.get[String]("slick.dbs.ragmeup.profile") match {
    case "slick.jdbc.SQLiteProfile$" => SQLiteProfile
    case "slick.jdbc.PostgresProfile$" => PostgresProfile
    case other =>
      throw new IllegalArgumentException(s"Unsupported database profile: $other")
  }

  import selectedProfile.api._

  class ChatLogTable(tag: Tag) extends Table[ChatLog](tag, "chat_logs") {
    def id = column[String]("id")
    def messageOffset = column[Int]("message_offset")
    def text = column[String]("text")
    def role = column[String]("role")
    def reply = column[String]("reply")
    def documents = column[String]("documents")
    def rewritten = column[Int]("rewritten")
    def question = column[String]("question")
    def fetchedNewDocuments = column[Int]("fetched_new_documents")

    def * = (id, messageOffset, text, role, reply, documents, rewritten == 1, question, fetchedNewDocuments == 1).mapTo[ChatLog]
  }
  val ChatLogs = TableQuery[ChatLogTable]

  class FeedbackTable(tag: Tag) extends Table[Feedback](tag, "feedback") {
    def id = column[String]("id")
    def messageOffset = column[Int]("message_offset")
    def feedback = column[Int]("feedback")

    def * = (id, messageOffset, feedback == 1).mapTo[Feedback]
  }
  val Feedbacks = TableQuery[FeedbackTable]
}
