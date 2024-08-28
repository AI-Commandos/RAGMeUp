name := """RAGMeUp"""
organization := "ai.commandos"

version := "1.0"

lazy val root = (project in file(".")).enablePlugins(PlayScala)

scalaVersion := "2.13.12"

libraryDependencies ++= Seq(
  guice,
  ws,
  "org.webjars" %% "webjars-play" % "3.0.1",
  "org.webjars" % "bootstrap" % "5.1.0",
  "org.webjars" % "jquery" % "3.6.0",
  "org.webjars" % "font-awesome" % "6.5.2",
  "com.nulab-inc" %% "scala-oauth2-core" % "1.6.0",
  "com.nulab-inc" %% "play2-oauth2-provider" % "2.0.0",
  "org.playframework" %% "play-slick" % "6.0.0",
  "org.playframework" %% "play-slick-evolutions" % "6.0.0",
  "com.mysql" % "mysql-connector-j" % "8.3.0",
  "com.github.t3hnar" %% "scala-bcrypt" % "4.3.0",
  "ai.x" %% "play-json-extensions" % "0.42.0",
  "org.scalatestplus.play" %% "scalatestplus-play" % "7.0.0" % Test
)
