// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
Pix2pix Edges2Pikachu example with p5.js using callback functions
This uses a pre-trained model on Pikachu images
For more models see: https://github.com/ml5js/ml5-data-and-training/tree/master/models/pix2pix
=== */

// The pre-trained Edges2Pikachu model is trained on 256x256 images
// So the input images can only be 256x256 or 512x512, or multiple of 256
const SIZE = 512;
let inputImg, inputCanvas, outputContainer, statusMsg, pix2pix, clearBtn, transferBtn, modelReady = false, isTransfering = false;
latestResult = null

function cityEngine(imageAsBase64 = latestResult) {
  encodedImage = encodeURI(imageAsBase64)
  $.post( "http://localhost:8887", { file_b64: encodedImage }, function( data ) {
    //setTimeout(function(){ $("#output_ce img").attr("src","tmp/" + data + "_out.png" ); }, 1000);
  });
}

function setup() {
  // Create a canvas
  inputCanvas = createCanvas(SIZE, SIZE);
  inputCanvas.class('border-box').parent('canvasContainer');

  // Display initial input image
  inputImg = loadImage('images/background.png', drawImage);

  // Selcect output div container
  outputContainer = select('#output');
  statusMsg = select('#status');

  // Select 'transfer' button html element
  transferBtn = select('#transferBtn');

  // Select 'clear' button html element
  clearBtn = select('#clearBtn');
  // Attach a mousePressed event to the 'clear' button
  clearBtn.mousePressed(function() {
    clearCanvas();
  });

    // Select 'clear' button html element
  downloadBtn = select('#downloadBtn');
  // Attach a mousePressed event to the 'clear' button
  downloadBtn.mousePressed(function(e) {
    cityEngine();
  });

    // Select 'clear' button html element
  imagineBtn = select('#imagineBtn');
  // Attach a mousePressed event to the 'clear' button
  imagineBtn.mousePressed(function() {
    tsvg()
  });

  drawRoadsBtn = select('#drawRoadsBtn');
  // Attach a mousePressed event to the 'clear' button
  drawRoadsBtn.mouseReleased(function() {
    $("#canvasContainer").css("z-index", 10)
    mode = CREATESTREETS;
    startX = 0;
    startY = 0;
    hideAnchors();

    $(editZoningBtn.elt).removeClass("selected");
    $(drawRoadsBtn.elt).addClass("selected");
  });

  editZoningBtn = select('#editZoningBtn');
  // Attach a mousePressed event to the 'clear' button
  editZoningBtn.mousePressed(function() {
    $("#canvasContainer").css("z-index", -1)
    mode = SELECT;
    showAnchors();
    $(drawRoadsBtn.elt).removeClass("selected");
    $(editZoningBtn.elt).addClass("selected");
  });

  

  // Set stroke to black
  //stroke(0, 0, 255);
  pixelDensity(1);
  //let red = color(255, 0, 0);
  //fill(red);

  stroke('blue');
  strokeWeight(2);

  // Create a pix2pix method with a pre-trained model
  pix2pix = ml5.pix2pix('models/citygen2-180.pict', modelLoaded);
}

newPointX = 0;
newPointY = 0;

polygons = [];
polygonPoints = [];

startX = 0;
startY = 0;

// Draw on the canvas when mouse is pressed
function draw() {
  /*if(mode == MOVEPOINT) {
    polygonPoints[selectedPoint][0] = mouseX;
    polygonPoints[selectedPoint][1] = mouseY;
  }*/
  if (mouseIsPressed) {
    if(startX == 0 && startY == 0) {
      startX = mouseX;
      startY = mouseY;
    }

    /*
    if(mode == SELECT && mouseX < 530 && mouseY < 530) {
      $img = $("#output img").css("z-index", -15)
      //letter.fill = 'red';
      two.update();
    }
    */
    /*
    match = false;
    for(i in polygonPoints) {
      p = polygonPoints[i];
      dist = Math.sqrt( (mouseX - p[0])**2 + (mouseY - p[1])**2 )
      if(dist < 5) {
        match = true;
        matchPoint = i;
        break;
      }
    }

    //line(mouseX, mouseY, pmouseX, pmouseY);
    //circle(mouseX, mouseY, 13);
    if(mode == SELECT) {
      if(match) {
        selectedPoint = matchPoint;
        mode = MOVEPOINT;
      }
      else {
        mode = NEWPOLYGON
        newPointX = mouseX;
        newPointY = mouseY;
      }
    }
    else if(mode == NEWPOLYGON) {
      if(match) {
        polygons.push(polygonPoints)
        polygonPoints = []
        mode = SELECT
      }
      if(newPointX == 0 && newPointY == 0) {
        newPointX = mouseX;
        newPointY = mouseY;
      }
    }
    */
  }
}

SELECT = 0
NEWPOLYGON = 1
MOVEPOINT = 2
CREATESTREETS = 3
mode = SELECT
selectedPoint = false;
dirty = true;

// Whenever mouse is released, transfer the current image if the model is loaded and it's not in the process of another transformation
function mouseReleased() {
  if(mode == CREATESTREETS && startX!=0 && startY !=0 ) {
    line(mouseX, mouseY, startX, startY);
    //tsvg();
    startX = 0;
    startY = 0;
  }
  /*

  for(i in polygonPoints) {
    p = polygonPoints[i];
    dist = Math.sqrt( (mouseX - p[0])**2 + (mouseY - p[1])**2 )
    if(dist < 5) {
      match = true;
      matchPoint = i;
      break;
    }
  }

  if(mode == NEWPOLYGON) {
      polygonPoints.push([newPointX, newPointY]);
    //circle(newPointX, newPointY, 5);
  }
  else if(mode == MOVEPOINT) {
    mode = SELECT;
    dirty = true;
  }

  background(0);


  beginShape();
  for(j in polygonPoints) {
    vertex(polygonPoints[j][0], polygonPoints[j][1]);
  }
  endShape(CLOSE);
  
  for(j in polygons) {
    curPolygonPoints = polygons[j]
    beginShape();
    for(i in curPolygonPoints) {
      vertex(curPolygonPoints[i][0], curPolygonPoints[i][1]);
    }
    endShape(CLOSE);
  }
  

  if(dirty) {
    setTimeout(function(){ transfer() }, 20);
    dirty = false;
  }

  newPointX = 0;
  newPointY = 0;

  //if (modelReady && !isTransfering) {
  //  transfer()
  //}
  */
}

// A function to be called when the models have loaded
function modelLoaded() {
  // Show 'Model Loaded!' message
  statusMsg.html('Neuroverkko ladattu!');

  // Set modelReady to true
  modelReady = true;

  //clearCanvas();
  // Call transfer function after the model is loaded
  //transfer();

  // Attach a mousePressed event to the transfer button
  /*transferBtn.mousePressed(function() {
    transfer();
  });*/
}

// Draw the input image to the canvas
function drawImage() {
  //image(inputImg, 0, 0);
}

// Clear the canvas
function clearCanvas() {
  polygonPoints = []
  
  clear();
  drawImage();
}

function tsvg() {
  var mySVG    = document.querySelector('#tempImageContainer svg'),      // Inline SVG element
    tgtImage = document.querySelector('#tempImage'),      // Where to draw the result
    can      = document.createElement('canvas'), // Not shown on page
    ctx      = can.getContext('2d'),
    loader   = new Image;                        // Not shown on page

  loader.width  = can.width  = 512;
  loader.height = can.height = 512;

  loader.onload = function(){
    ctx.drawImage( document.querySelector('#bgContainer img'), 0, 0, 512, 512);
    ctx.drawImage( loader, 0, 0, loader.width, loader.height );
    ctx.drawImage( document.querySelector('#canvasContainer canvas'), 0, 0, 512, 512);
    tgtImage.src = can.toDataURL();
    setTimeout(function(){ transfer(select('#tempImage').elt); }, 150);
  };

  var svgAsXML = (new XMLSerializer).serializeToString( mySVG );
  loader.src = 'data:image/svg+xml,' + encodeURIComponent( svgAsXML );

}

function transfer(canvasElement, cb) {
  if(!canvasElement)
    canvasElement = select('canvas').elt;

  // Set isTransfering to true
  isTransfering = true;

  // Update status message
  statusMsg.html('Applying Style Transfer...!');

  // Select canvas DOM element

  // Apply pix2pix transformation
  pix2pix.transfer(canvasElement, function(err, result) {
    if (err) {
      console.log(err);
    }
    if (result && result.src) {
      // Set isTransfering back to false
      isTransfering = false;
      // Clear output container
      outputContainer.html('');
      //$("#downloadBtn a")[0].href=result.src;
      $("#downloadBtn").css('opacity', 1)
      // Create an image based result
      latestResult = result.src
      //createImg(result.src).class('border-box').parent('output');

      //letter.fill = '#ff040436';
      //two.update();

      //__img = $("#output img")[0]
      //__img.zIndex = 100001
      /*
      $img.on("error", function() {
        setTimeout(function(){ 
          src = $img.attr("src")
          if(src[src.length-1] != "?")
            src = src + "?"
          src = src + "_"
          $img.attr("src", src); 
        }, 1000);
      })*/

      //cityEngine();
      // Show 'Done!' message
      //statusMsg.html('Done!');

      if(cb) cb(result);
    }
  });
}
