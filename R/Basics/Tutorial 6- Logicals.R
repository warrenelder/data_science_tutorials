
for(i in 1:10){
  var[i] <- rnorm(1)
  if(var[i] > 1){
    print("greater than 1")
  } else if(var[i] == 1) {
    print("equal 1")
  } else {
    print("less than one")
  }
}
