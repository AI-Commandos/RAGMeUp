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
  "com.github.t3hnar" %% "scala-bcrypt" % "4.3.0",
  "ai.x" %% "play-json-extensions" % "0.42.0",
  "org.playframework" %% "play-slick" % "6.1.1",
  "org.playframework" %% "play-slick-evolutions" % "6.1.1",
  "com.typesafe.slick" %% "slick" % "3.5.2",
  "com.typesafe.slick" %% "slick-hikaricp" % "3.5.2",
  "org.xerial" % "sqlite-jdbc" % "3.47.0.0",
  "org.postgresql" % "postgresql" % "42.7.4",
  "org.scalatestplus.play" %% "scalatestplus-play" % "7.0.0" % Test
)
