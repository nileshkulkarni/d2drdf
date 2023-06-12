// from http://bl.ocks.org/mbostock/4349187
// Sample from a normal distribution with mean 0, stddev 1.

function setup_slider(seg_type){
  var slider = document.getElementById('slider-'+seg_type);
  var label1 = document.getElementById('slider-label1-'+seg_type);
  var label1Text = document.getElementById('slider-label1-text-'+seg_type);
  var label2 = document.getElementById('slider-label2-'+seg_type);
  var label2Text = document.getElementById('slider-label2-text-'+seg_type);
  var irange = [0, 3]
  label1.style.margin = "10px"
  label2.style.margin = "-30px"
  noUiSlider.create(slider, {
    start: [1., 2.], // Initial positions of the sliders
    connect: true, // Connect the sliders
    range: {
      'min': irange[0],
      'max': irange[1]
    }
  });
  // Update labels on slider update
  slider.noUiSlider.on('update', function (values, handle) {
    var value = values[handle];
    var label = handle === 0 ? label1Text : label2Text;
    if (seg_type == 'ii'){
      label.textContent = 'I' + (handle + 1) +  ': ' + value;
    }
    else if (seg_type == 'oi'){
      if (handle==0){
        label.textContent = '0'  +  ': ' + value;
      }
      else{
        label.textContent = 'I'  +  ': ' + value;
      }
    }
    else if (seg_type == 'io'){
      if (handle==0){
        label.textContent = 'I'  +  ': ' + value;
      }
      else{
        label.textContent = 'O'  +  ': ' + value;
      }
    }
    else if (seg_type == 'oo'){
      label.textContent = 'O' + (handle + 1) +  ': ' + value;
    }
   
    // label.style.position= 'abolute'
    var translateX = Math.round(100*value/irange[1]).toString() + "%"; // New translateX value as a percentage
    var translateY = "50%"; 
    label.style.transform = "translateX(" + translateX + ")";
    draw_example(values[0], values[1], seg_type)
    
  });
}


function updateSliderValues(i1, i2,seg_type) {
  var slider = document.getElementById('slider-'+seg_type);
  var newValues = [i1, i2]; // New values for the sliders

  slider.noUiSlider.set(newValues); // Update the slider values
}


function gtDRDF (x, I1, I2) {
  if (Math.abs(I1 -x) < Math.abs(I2 -x)){
    return (I1 - x);
  }
  else{
    return  (I2 - x);
  }
  // return (I1 - x) ? Math.abs(I1 - x) < Math.abs(I2 - x) : (I2 - x)
}

function iiValue(x, y, I1, I2) {
  return Math.abs(y - gtDRDF(x, I1, I2))
}

function generate_ii_data(x, y, i1, i2,){
  var data = [];
  var ncolumns = x.length;
  var nrows = y.length;
  for (var i = 0; i < nrows; i++) {
    var row = [];
    for (var j = 0; j < ncolumns; j++) {
     
      // var value = Math.sin(x) + Math.cos(y); // Example function to generate values
      var value = iiValue(x[j], y[i], i1, i2); // Example function to generate values
      // var value = y[i]
      row.push(value);
    }
    data.push(row);
  }
  return data;
}

function ubOI(s, x){
  return s - x
}

function gtOI(e, x){
  return e - x
}

function oiValue(x, y, s, e) {
  u = ubOI(s,x);
  sCloser = Math.abs(s - x) < Math.abs(e - x);
  if (sCloser){
    return Math.min(Math.max(0, y - u), Math.abs(y - gtOI(e, x)));
  }
  else{
    return Math.abs(y - gtOI(e, x));
  }
}

function generate_oi_data(x, y, i1, i2,){
  var data = [];
  var ncolumns = x.length;
  var nrows = y.length;
  for (var i = 0; i < nrows; i++) {
    var row = [];
    for (var j = 0; j < ncolumns; j++) {
     
      // var value = Math.sin(x) + Math.cos(y); // Example function to generate values
      var value = oiValue(x[j], y[i], i1, i2); // Example function to generate values
      // var value = y[i]
      row.push(value);
    }
    data.push(row);
  }
  return data;
}


function lbIO(e, x){
  return e - x;
}

function gtIO(s, x){
  return s - x
}

function ioValue(x, y, s, e) {
  l = lbIO(e, x)
  sCloser = Math.abs(s - x) < Math.abs(e - x);
  if (sCloser){
    return Math.abs(y - gtIO(s, x))
  }
  else{
    return Math.min(Math.max(0, l - y), Math.abs(y - gtIO(s, x)))
  }
}
function generate_io_data(x, y, i1, i2,){
  var data = [];
  var ncolumns = x.length;
  var nrows = y.length;
  for (var i = 0; i < nrows; i++) {
    var row = [];
    for (var j = 0; j < ncolumns; j++) {
     
      // var value = Math.sin(x) + Math.cos(y); // Example function to generate values
      var value = ioValue(x[j], y[i], i1, i2); // Example function to generate values
      // var value = y[i]
      row.push(value);
    }
    data.push(row);
  }
  return data;
}

function ubOO(s,x){
  return s - x;
}

function lbOO(e, x){
  return e - x;
}

function ooValue(x, y, s, e) {
  var u = ubOO(s, x);
  var l = lbOO(e, x);
  return Math.max(0, (l - u) / 2 - Math.abs(y - (u + l) / 2))
}
function generate_oo_data(x, y, i1, i2,){
  var data = [];
  var ncolumns = x.length;
  var nrows = y.length;
  for (var i = 0; i < nrows; i++) {
    var row = [];
    for (var j = 0; j < ncolumns; j++) {
     
      // var value = Math.sin(x) + Math.cos(y); // Example function to generate values
      var value = ooValue(x[j], y[i], i1, i2); // Example function to generate values
      // var value = y[i]
      row.push(value);
    }
    data.push(row);
  }
  return data;
}

function draw_example(value1, value2, seg_type) {
  let freq = 5
  let i1 = value1;
  let i2 = value2;

  var nrows = 100
  var ncols = 100

  var xrange = [0, 3]
  var yrange = [-1, 1]

  var emptyArray = []


  var xvaluesArray = d3.range(xrange[0], xrange[1], (xrange[1] - xrange[0]) / nrows);
  var yvaluesArray = d3.range(yrange[0], yrange[1], (yrange[1] - yrange[0]) / ncols);

  if (seg_type == 'ii'){
    penalty = generate_ii_data(xvaluesArray, yvaluesArray, i1, i2)
    var E1Text = 'I1';
    var E2Text = 'I2';
    var E1Color = '#648FFF';
    var E2Color = '#648FFF';
    var title = 'II Segment Penalty Plot';
  }
  else if (seg_type == 'io'){
    penalty = generate_io_data(xvaluesArray, yvaluesArray, i1, i2)
    var E1Text = 'I ';
    var E2Text = 'O';
    var E1Color = '#648FFF';
    var E2Color = '#C22DD5';
    var title = 'IO Segment Penalty Plot';

  }
  else if (seg_type == 'oi'){
    penalty = generate_oi_data(xvaluesArray, yvaluesArray, i1, i2)
    var E1Text = 'O';
    var E2Text = 'I ';
    var E1Color = '#C22DD5';
    var E2Color = '#648FFF';
    var title = 'OI Segment Penalty Plot';

  }
  else if (seg_type == 'oo'){
    penalty = generate_oo_data(xvaluesArray, yvaluesArray, i1, i2)
    var E1Text = 'O1';
    var E2Text = 'O2';
    var E1Color = '#C22DD5';
    var E2Color = '#C22DD5';
    var title = 'OO Segment Penalty Plot';
  }
  else{
    console.log('wrong segment type')
  }
  
  
  var I1trace = {
    x: [i1, i1, i1],
    y: [-1, 0, 1],
    mode: 'lines+text',
    textposition: 'left',
    line: {
      color: 'rgb(0, 0, 0)',
      width: 3
    },
    
  }
  var I2trace = {
    x: [i2, i2, i2],
    y: [-1, 0, 1],
    mode: 'lines+text',
    // text: ['', 'I2', ''],
    textposition: 'right',
    line: {
      color: 'rgb(0, 0, 0)',
      width: 3
    },
    
  }

  var penaltydata = {
    z: penalty,
    x: xvaluesArray,
    y: yvaluesArray,
    type: 'heatmap',
    colorscale: 'Reds',
    showscale: false,
    colorbar: {
      visible: false
    }
  };

  // console.log(freq / 10)
  // var data = [trace3, marker1];
  var data = [I1trace, I2trace, penaltydata]
  // var data = [penaltydata]

  var layout = {
    width: 500,
    height: 500,
    title: title,
    yaxis: {
      range: [-1.0, 1.1],
      zeroline: false
    },
    xaxis: {
      range: [0, 3],
      zeroline: false
    },
    showlegend: false,
    annotations: [
      {
        x: i1,
        y: 0,
        xref: 'x',
        yref: 'y',
        text: E1Text,
        showarrow: false,
        font: {
          size: 30,
          color: E1Color
        },
        align: 'center',
        bgcolor: 'white',
        bordercolor: 'black',
        borderwidth: 2,
        xanchor: 'center',
        yanchor: 'middle',
        xshift: -0,
        yshift: 0
      },
      {
        x: i2,
        y: 0,
        xref: 'x',
        yref: 'y',
        text: E2Text,
        showarrow: false,
        font: {
          size: 30,
          color: E2Color
        },
        align: 'center',
        bgcolor: 'white',
        bordercolor: 'black',
        borderwidth: 2,
        xanchor: 'center',
        yanchor: 'middle',
        xshift: 0,
        yshift: 0
      }
    ]
  }
  var config = {
    displayModeBar: false
  };
  Plotly.newPlot('penalty-'+seg_type, data, layout, config)
}
