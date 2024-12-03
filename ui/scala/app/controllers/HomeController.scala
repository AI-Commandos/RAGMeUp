package controllers

import javax.inject._
import play.api._
import play.api.libs.json._
import play.api.mvc._
import play.api.libs.ws._

import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.duration._
import scala.language.postfixOps
import scala.concurrent.{ExecutionContext, Future}

@Singleton
class HomeController @Inject()(
  cc: ControllerComponents,
  config: Configuration,
  ws: WSClient
)(implicit ec: ExecutionContext) // Use the injected ExecutionContext
  extends AbstractController(cc) {

  def index() = Action { implicit request: Request[AnyContent] =>
    Ok("Index page")
  }

  def asyncExample() = Action.async { implicit request: Request[AnyContent] =>
    Future.successful(Ok("Async example"))
  }

  def add() = Action { implicit request: Request[AnyContent] =>
    Ok("Add page")
  }

  def download(file: String) = Action { implicit request: Request[AnyContent] =>
    Ok(s"Download page for file: $file")
  }

  def delete(file: String) = Action { implicit request: Request[AnyContent] =>
    Ok(s"Delete page for file: $file")
  }

  def upload() = Action { implicit request: Request[AnyContent] =>
    Ok("Upload page")
  }

  def feedback() = Action { implicit request: Request[AnyContent] =>
    Ok("Feedback page")
  }

  def searchGraph(query: String, docs: Seq[JsObject])(implicit ec: ExecutionContext): Future[String] = {
    Future {
      "response from graph" // Replace with your logic
    }
  }

  def search() = Action.async { implicit request: Request[AnyContent] =>
    val json = request.body.asJson.getOrElse(Json.obj()).as[JsObject]
    val query = (json \ "query").as[String]
    val history = (json \ "history").as[Seq[JsObject]]
    val docs = (json \ "docs").as[Seq[JsObject]]

    ws.url(s"${config.get[String]("server_url")}/chat")
      .withRequestTimeout(5.minutes)
      .post(Json.obj(
        "prompt" -> query,
        "history" -> history,
        "docs" -> docs
      ))
      .map(response =>
        Ok(response.json)
      )
  }
}
