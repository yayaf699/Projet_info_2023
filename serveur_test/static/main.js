const canvas = document.getElementById("canvas");
canvas.width = window.innerWidth -60;
canvas.height = 400;


let context = canvas.getContext("2d");
start_background_color = "white";
context.fillStyle = start_background_color;
context.fillRect(0, 0, canvas.width, canvas.height);

let draw_color = "black";
let draw_width = "10";
let is_drawing = false;

let tab_dessin = [];
let index = -1; 


function change_color(element){
    draw_color = element.style.background;
}

canvas.addEventListener("touchstart", start, false);
canvas.addEventListener("touchmove", draw, false);
canvas.addEventListener("mousedown", start, false);
canvas.addEventListener("mousemove", draw, false);

canvas.addEventListener("touchend", stop, false);
canvas.addEventListener("mouseup", stop, false);
canvas.addEventListener("mouseout", stop, false);



function start(event){
    is_drawing = true;
    context.beginPath();
    context.moveTo(event.clientX - canvas.offsetLeft, 
                   event.clientY-canvas.offsetTop);
    event.preventDefault();
}

function draw(event){
    if (is_drawing)
    {
        context.lineTo(event.clientX - canvas.offsetLeft, 
            event.clientY-canvas.offsetTop);
        context.strokeStyle = draw_color;
        context.lineWidth = draw_width;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.stroke();
    }
    event.preventDefault();
}


function stop(event){
    if (is_drawing)
    {
        context.stroke();
        context.closePath();
        is_drawing = false;

    }
    event.preventDefault();
    
    if (event.type != 'mouseout'){
        tab_dessin.push(context.getImageData(0, 0, canvas.width, canvas.height));
        index +=1;
    }
}

function clear_canvas(){
    context.fillStyle = start_background_color;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillRect(0, 0, canvas.width, canvas.height);

    tab_dessin = [];
    index = -1;

}

function undo(){

    if (index <= 0){
        clear_canvas();
    }
    else{
        index -= 1;
        tab_dessin.pop();
        context.putImageData(tab_dessin[index], 0,0);
    }
     
}

function save(){

    const canvas = document.getElementById("canvas");

    // Récupération de l'image base64
    const data = canvas.toDataURL();

    // Création d'un formulaire avec l'image en tant que donnée
    const form = document.createElement("form");
    form.method = "POST";
    form.action = "/home";
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = "image";
    input.value = data;
    form.appendChild(input);

    // Envoi du formulaire
    document.body.appendChild(form);

    form.submit();

    //clear 
    clear_canvas();

}