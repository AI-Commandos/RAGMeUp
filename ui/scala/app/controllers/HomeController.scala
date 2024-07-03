package controllers

import javax.inject._
import play.api._

import java.nio.file.{FileSystems, Files, Paths}
import play.api.libs.json._
import play.api.mvc._
import play.api.libs.ws._

import scala.concurrent.duration.{Duration, DurationInt}
import scala.concurrent.{ExecutionContext, Future}
import scala.jdk.CollectionConverters._
import scala.language.postfixOps
import scala.util.Try

@Singleton
class HomeController @Inject()(
    cc: ControllerComponents,
    config: Configuration,
    ws: WSClient
) (implicit ec: ExecutionContext) extends AbstractController(cc) {

  def index() = Action { implicit request: Request[AnyContent] =>
    Ok(views.html.index(config))
  }

  def add() = Action { implicit request: Request[AnyContent] =>
    val files = Try {
      Files.list(FileSystems.getDefault.getPath(config.get[String]("data_folder"))).iterator().asScala.map(_.getFileName.toString).toSeq
    } getOrElse(Nil)
    Ok(views.html.add(files))
  }

  def search() = Action.async { implicit request: Request[AnyContent] =>
    val json = request.body.asJson.getOrElse(Json.obj()).as[JsObject]
    val query = (json \ "query").as[String]
    val history = (json \ "history").as[Seq[JsObject]]
    val docs = (json \ "docs").as[Seq[JsObject]]

    ws
      .url(s"${config.get[String]("server_url")}/chat")
      .withRequestTimeout(5 minutes)
      .post(Json.obj(
        "prompt" -> query,
        "history" -> history,
        "docs" -> docs
      ))
      .map(response =>
          Ok(response.json)
      )
  }

  def download(file: String) = Action { implicit request: Request[AnyContent] =>
    val filePath = new java.io.File(s"${config.get[String]("data_folder")}/$file")

    if (filePath.exists && filePath.isFile) {
      Ok.sendFile(
        content = filePath,
        fileName = _ => Some(file),
        inline = false
      )
    } else {
      NotFound(s"File '$file' not found.")
    }
  }

  def upload = Action(parse.multipartFormData) { implicit request =>
    request.body.file("file").map { file =>
      val filename = Paths.get(file.filename).getFileName
      val dataFolder = config.get[String]("data_folder")
      val filePath = new java.io.File(s"$dataFolder/$filename")

      file.ref.copyTo(filePath)

      Redirect(routes.HomeController.add()).flashing("success" -> "Added CV to the database.")
    }.getOrElse {
      Redirect(routes.HomeController.add()).flashing("error" -> "Adding CV to database failed.")
    }
  }
}
