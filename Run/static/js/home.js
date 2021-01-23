// $(document).ready(function(curUsers){

// 	console.log(curUsers)

// 	showExistingUsers(curUsers)
// });
	function showExistingUsers(curUsers){
		console.log("sulooood" + curUsers)
		// mockUsers = ["LincyLegada", "ShebnaRoseFabilloren"]
		console.log("USERS" + curUsers)
		localHostURL = "http://localhost:5000/uploadtest/"



		for (var i = 0; i < curUsers.length; i++) {
			var name = curUsers[i]
			var classes = "btn btn-block waves-effect waves-light"

			var link = $("<a class='collection-item'>" + name + "</a>").attr("href", localHostURL + name).attr("class", classes)

			$("#user-list").append(link)
			$("#user-list").slideDown()
		}
	}
