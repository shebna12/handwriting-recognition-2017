	function check(){
		var resultChar = words.split()
		var correctChar = correct.split()
		var stoppedAt = 0

		for (var i = 0; i < resultChar.length && i < correctChar.length ; i++) {
			var letter = resultChar[i]
			var corret_letter = correctChar[i]
			var span = $("<span>" + (resultChar[i] + "</span>")
			
			if (letter == correct_letter) {
				span.attr("class", "right")
			} else {
				span.attr("class", "wrong")
			}

			$("#results").append(span)

			stoppedAt = i
		}

		if (stoppedAt < resultChar.length) {
			for (var i = stoppedAt; i < resultChar.length; i++) {
				var span = $("<span>" + (resultChar[i] + "</span>")				
				span.attr("class", "wrong")
				$("#results").append(span)
			}
		} else if (stoppedAt < correctChar.length) {
			for (var i = stoppedAt; i < correctChar.length; i++) {
				var span = $("<span>&nbsp;</span>")				
				span.attr("class", "wrong")
				$("#results").append(span)
			}
		}

	}