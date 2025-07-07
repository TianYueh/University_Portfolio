function test() {
    var input = document.getElementById("input").value;
    document.write("你想查詢的地方是：" + input);
    document.write("<br>Successful!");
}

var geocoder = new google.maps.Geocoder();

function codeAddress() {
    var address = document.getElementById("input").value;
    console.log("你想查詢的地方是：" + address);
 
    geocoder.geocode( { 'address': address}, function(results, status) {
      if (status == google.maps.GeocoderStatus.OK) {   
        console.log("<br>Successful!");
		var lat=results[0].geometry.location.lat();
		var lng=results[0].geometry.location.lng();
        alert(lat+"\n"+lng);
    }});
}


async function callapi() {
    const response = await fetch("https://udxas0gx4i.execute-api.us-east-1.amazonaws.com/dev/",{method:"POST"})
    console.log(response)
    const data = await response.json()
    console.log(data)
}





