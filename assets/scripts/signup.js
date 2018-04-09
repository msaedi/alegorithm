console.log("Hello")

function register(username, password, name, email)
{
    console.log("In register function")
    console.log("username="+username)
    console.log("password="+password)
    console.log("name="+name)
    console.log("email="+email)
    url = "register?username="+username+"&password="+password+"&name="+name+"&email="+email
    console.log(url)
    d3.tsv(
        url,
        function(d)
        {
		console.log(d[0].msg)
		console.log(d[0].username)
		if (d[0].msg == 'user added')
		{
			console.log('was success')
			window.location.href = "signin.html";
		}
	});
}

console.log("Registration")
var email = document.getElementById("email-header15-w");
var username = document.getElementById("username-header15-w");
var password = document.getElementById("password-header15-w");
var fullname = document.getElementById("fullname-header15-w");
console.log(fullname.value)
d3.select("#signup").on("click", function() {register(username.value, password.value, fullname.value, email.value)})
console.log("End of Registration")