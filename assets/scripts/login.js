console.log("Hello")

function authenticate(username, password)
{
    console.log("In function")
    url = "login?username="+username+"&password="+password
    console.log(url)
    d3.tsv(
        url,
        function(d)
        {
		console.log(d[0].msg)
		console.log(d[0].name)
		if (d[0].msg == 'success')
		{
			console.log('was success')
			//console.log(document.getElementById("signin_button").text)
			//console.log(document.getElementById("signin_button").href)
			//document.getElementById("signin_button").disabled = true
			window.location.href = '/'
		}
	});
}

console.log("Start")
var username = document.getElementById("username-header15-w");
var password = document.getElementById("password-header15-w");
d3.select("#signin").on("click", function() {authenticate(username.value, password.value)})