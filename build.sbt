name := "scala"

version := "0.1"

scalaVersion := "2.12.6"

libraryDependencies ++= Seq(
  // https://mvnrepository.com/artifact/org.scalanlp/breeze
  "org.scalanlp" %% "breeze" % "1.0-RC2",
  // https://mvnrepository.com/artifact/org.scalanlp/breeze-natives
  "org.scalanlp" %% "breeze-natives" % "1.0-RC2",
  // https://mvnrepository.com/artifact/org.scalanlp/breeze-viz
  "org.scalanlp" %% "breeze-viz" % "1.0-RC2"
)


