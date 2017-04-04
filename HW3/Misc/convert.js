const translate = require('google-translate-api');
fs = require('fs')

const file = process.argv[2];


fs.readFile(file, 'utf8', function (err,data) {
  if (err) {
  	console.log("File i/o error:");
    console.log(err);
  }
  const allLines = data.split(/\r\n|\n/);
  // Reading line by line
  allLines.map((line) => {
    translate(line, {to: 'hi'}).then(res => {
    		console.log(res.text);
		}).catch(err => {
   			console.error("Translation error:");
    		console.error(err);
		});
    });
});
