name := "LassoMLLib"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVers = "2.0.0"

libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value

libraryDependencies += "org.scalaz" %% "scalaz-core" % "7.2.5"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.10
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0" % "provided"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib_2.10
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.0" % "provided"

