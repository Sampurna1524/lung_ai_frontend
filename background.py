import streamlit as st
import streamlit.components.v1 as components


def load_background():

    st.markdown("""
    <style>

    html, body {
    background:#020617 !important;
    }

    .stApp{
    background:transparent !important;
    }

    [data-testid="stAppViewContainer"]{
    background:transparent !important;
    }

    [data-testid="stHeader"]{
    background:transparent !important;
    }


    /* ---------- REMOVE SIDEBAR ---------- */

[data-testid="stSidebar"]{
    display:none !important;
}

section[data-testid="stSidebar"]{
    display:none !important;
    width:0px !important;
    min-width:0px !important;
    max-width:0px !important;
}

[data-testid="collapsedControl"]{
    display:none !important;
}

[data-testid="stSidebarNav"]{
    display:none !important;
}

    iframe{
    position:fixed !important;
    top:0;
    left:0;
    width:100vw !important;
    height:100vh !important;
    border:none !important;
    z-index:-1 !important;
    }

    </style>
    """, unsafe_allow_html=True)


    background = """
    <!DOCTYPE html>
    <html>
    <head>

    <style>

    html,body{
    margin:0;
    overflow:hidden;
    background:#020617;
    }

    #particles-js{
    position:fixed;
    width:100%;
    height:100%;
    top:0;
    left:0;
    z-index:-2;
    }

    #hologram{
    position:fixed;
    top:50%;
    left:50%;
    transform:translate(-50%,-50%);
    width:900px;
    height:900px;
    opacity:0.35;
    pointer-events:none;
    }

    </style>

    </head>

    <body>

    <div id="particles-js"></div>
    <div id="hologram"></div>

    <script src="https://cdn.jsdelivr.net/npm/particles.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/loaders/GLTFLoader.js"></script>

    <script>

    particlesJS("particles-js", {
    "particles":{
    "number":{"value":80},
    "color":{"value":"#00E0FF"},
    "shape":{"type":"circle"},
    "opacity":{"value":0.5},
    "size":{"value":3},
    "line_linked":{
    "enable":true,
    "distance":150,
    "color":"#00E0FF",
    "opacity":0.3,
    "width":1
    },
    "move":{"enable":true,"speed":2}
    }
    });


    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45,1,0.1,1000);

    const renderer = new THREE.WebGLRenderer({
    alpha:true,
    antialias:true
    });

    renderer.setSize(900,900);

    document.getElementById("hologram").appendChild(renderer.domElement);


    const light = new THREE.PointLight(0x00ffff,1.5);
    light.position.set(10,10,10);
    scene.add(light);

    scene.add(new THREE.AmbientLight(0x404040));


    const loader = new THREE.GLTFLoader();
    let torso;

    loader.load(
    "https://modelviewer.dev/shared-assets/models/Astronaut.glb",
    function(gltf){

    torso = gltf.scene;

    torso.scale.set(5,5,5);

    const box = new THREE.Box3().setFromObject(torso);
    const center = box.getCenter(new THREE.Vector3());

    torso.position.x += (torso.position.x - center.x);
    torso.position.y += (torso.position.y - center.y);
    torso.position.z += (torso.position.z - center.z);

    torso.position.y -= 1.5;

    torso.traverse(function(child){

    if(child.isMesh){

    child.material = new THREE.MeshBasicMaterial({
    color:0x00ffff,
    wireframe:true,
    transparent:true,
    opacity:0.35
    });

    }

    });

    scene.add(torso);

    }
    );


    camera.position.set(0,0,8);


    let mouseX = 0;
    let mouseY = 0;

    window.parent.addEventListener("mousemove",(event)=>{

    mouseX = (event.clientX/window.innerWidth)*2-1;
    mouseY = -(event.clientY/window.innerHeight)*2+1;

    });


    function animate(){

    requestAnimationFrame(animate);

    if(torso){

    torso.rotation.x = mouseY * 0.3;
    torso.rotation.y = mouseX * 0.3;

    }

    renderer.render(scene,camera);

    }

    animate();

    </script>

    </body>
    </html>
    """

    components.html(background, height=0)