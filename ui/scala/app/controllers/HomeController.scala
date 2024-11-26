package controllers

import org.apache.pekko.stream.scaladsl.{FileIO, Source}

import javax.inject._
import play.api._
import play.api.http.HttpEntity

import java.nio.file.{Files, Paths}
import play.api.libs.json._
import play.api.mvc._
import play.api.libs.ws._
import play.api.mvc.MultipartFormData.{DataPart, FilePart}

import java.io.File
import scala.concurrent.duration.DurationInt
import scala.concurrent.{ExecutionContext, Future}
import scala.language.postfixOps

@Singleton
class HomeController @Inject()(
    cc: ControllerComponents,
    config: Configuration,
    ws: WSClient
) (implicit ec: ExecutionContext) extends AbstractController(cc) {

  def index() = Action { implicit request: Request[AnyContent] =>
    Ok(views.html.index(config))
  }

  def add() = Action.async { implicit request: Request[AnyContent] =>
    ws
      .url(s"${config.get[String]("server_url")}/get_documents")
      .withRequestTimeout(5 minutes)
      .get()
      .map(files => {
        Ok(views.html.add(files.json.as[Seq[String]]))
      })
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

  def download(file: String) = Action.async { implicit request: Request[AnyContent] =>
    ws.url(s"${config.get[String]("server_url")}/get_document")
      .withRequestTimeout(5.minutes)
      .post(Json.obj("filename" -> file))
      .map { response =>
        if (response.status == 200) {
          // Get the content type and filename from headers
          val contentType = response.header("Content-Type").getOrElse("application/octet-stream")
          val disposition = response.header("Content-Disposition").getOrElse("")
          val filenameRegex = """filename="?(.+)"?""".r
          val downloadFilename = filenameRegex.findFirstMatchIn(disposition).map(_.group(1)).getOrElse(file)

          // Stream the response body to the user
          Result(
            header = ResponseHeader(200, Map(
              "Content-Disposition" -> s"""attachment; filename="$downloadFilename"""",
              "Content-Type" -> contentType
            )),
            body = HttpEntity.Streamed(
              response.bodyAsSource,
              response.header("Content-Length").map(_.toLong),
              Some(contentType)
            )
          )
        } else {
          // Handle error cases
          Status(response.status)(s"Error: ${response.statusText}")
        }
      }
  }

  def upload = Action.async(parse.multipartFormData) { implicit request =>
    request.body.file("file").map { file =>
    // Copy over file
    val filename = Paths.get(file.filename).getFileName
    val dataFolder = config.get[String]("data_folder")
    val filePath = new java.io.File(s"$dataFolder/$filename")

    // Create folder if it doesn't exist yet
    val dataFolderFile = new File(dataFolder)
    if (!dataFolderFile.exists()) {
      if (dataFolderFile.mkdirs()) {} else {
        throw new RuntimeException(s"Failed to create directory $dataFolder.")
      }
    } else if (!dataFolderFile.isDirectory) {
      throw new RuntimeException(s"$dataFolder exists but is not a directory.")
    }

    file.ref.copyTo(filePath)

    // Prepare the file as a FilePart
    val filePart = FilePart(
      key = "file",
      filename = filePath.getName,
      contentType = Some(Files.probeContentType(filePath.toPath)),
      ref = FileIO.fromPath(filePath.toPath)
    )

    // Send the file as multipart/form-data
    ws.url(s"${config.get[String]("server_url")}/add_document")
      .withRequestTimeout(5.minutes)
      .post(Source(List(filePart)))
      .map { response =>
        // Remove the file locally
        filePath.delete()

        response.status match {
          case OK =>
            Redirect(routes.HomeController.add()).flashing("success" -> "Added file to the database.")
          case _ => Redirect(routes.HomeController.add()).flashing("error" -> "Adding file to database failed.")
        }
      }.recover {
        case e: Exception => {
          filePath.delete()
          Redirect(routes.HomeController.add()).flashing("error" -> s"Internal server error: ${e.getMessage()}")
        }
      }
    }.getOrElse {
      Future.successful(Redirect(routes.HomeController.add()).flashing(
        "error" -> "No file found to upload."
      ))
    }
  }

  def delete(file: String) = Action.async { implicit request =>
    ws.url(s"${config.get[String]("server_url")}/delete")
      .withRequestTimeout(5.minutes)
      .post(Json.obj("filename" -> file))
      .map { response =>
        val deleteCount = (response.json.as[JsObject] \ "count").as[Int]
        Redirect(routes.HomeController.add())
          .flashing("success" -> s"File ${file} has been deleted (${deleteCount} chunks in total).")
      }
  }

  def feedback() = Action { implicit request: Request[AnyContent] =>
    Ok(Json.obj())
  }
}
