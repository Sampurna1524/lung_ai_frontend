import streamlit as st
import streamlit.components.v1 as components
# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Lung AI Login",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="🫁"
)

# =========================================
# SESSION STATE
# =========================================

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "username" not in st.session_state:
    st.session_state["username"] = ""

# =========================================
# REDIRECT IF LOGGED IN
# =========================================

if st.session_state["logged_in"]:
    st.switch_page("pages/Home.py")

# =========================================
# STYLING
# =========================================

st.markdown(
    """
    <style>

    /* HIDE STREAMLIT DEFAULT UI */

    [data-testid="stSidebar"]{
        display:none !important;
    }

    [data-testid="collapsedControl"]{
        display:none !important;
    }

    [data-testid="stToolbar"]{
        display:none !important;
    }

    header{
        visibility:hidden;
    }

    footer{
        visibility:hidden;
    }

    /* PAGE */

    html, body, [data-testid="stAppViewContainer"]{
        background:#020617 !important;
        overflow-x:hidden !important;
    }

    .stApp{
        background:#020617;
    }

    .main{
    position:relative;
    z-index:2;
}

    /* =========================================
   ANIMATED BACKGROUND
========================================= */

[data-testid="stAppViewContainer"]{

    background:
        radial-gradient(circle at top left,
        rgba(0,212,255,0.12),
        transparent 28%),

        radial-gradient(circle at bottom right,
        rgba(108,99,255,0.12),
        transparent 30%),

        #020617;

    overflow-x:hidden;
}

/* GRID */

[data-testid="stAppViewContainer"]::before{

    content:"";

    position:fixed;

    inset:0;

    background-image:
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);

    background-size:42px 42px;

    mask-image:linear-gradient(to bottom,
        rgba(0,0,0,0.7),
        transparent);

    pointer-events:none;

    z-index:-2;
}

/* =========================================
   PREMIUM ANIMATED BACKGROUND
========================================= */

.orb{

    position:fixed;

    border-radius:50%;

    pointer-events:none;

    z-index:0;

    will-change:transform;

    mix-blend-mode:screen;

    animation:floatOrb 18s ease-in-out infinite;
}

/* MAIN CYAN GLOW */

.orb1{

    width:700px;
    height:700px;

    background:
        radial-gradient(circle,
        rgba(0,224,255,0.34),
        rgba(0,224,255,0.08),
        transparent 72%);

    top:-220px;
    left:-220px;

    animation-delay:0s;
}

/* MAIN PURPLE GLOW */

.orb2{

    width:650px;
    height:650px;

    background:
        radial-gradient(circle,
        rgba(108,99,255,0.30),
        rgba(108,99,255,0.08),
        transparent 72%);

    bottom:-240px;
    right:-220px;

    animation-delay:3s;
}

/* CENTER AMBIENT LIGHT */

.orb3{

    width:420px;
    height:420px;

    background:
        radial-gradient(circle,
        rgba(0,234,255,0.18),
        transparent 75%);

    top:35%;
    left:10%;

    animation-delay:6s;
}

/* TOP RIGHT LIGHT */

.orb4{

    width:320px;
    height:320px;

    background:
        radial-gradient(circle,
        rgba(139,92,246,0.16),
        transparent 75%);

    top:6%;
    right:10%;

    animation-delay:9s;
}

/* LOWER CENTER LIGHT */

.orb5{

    width:280px;
    height:280px;

    background:
        radial-gradient(circle,
        rgba(34,211,238,0.16),
        transparent 75%);

    bottom:8%;
    left:40%;

    animation-delay:12s;
}

/* SMOOTH FLOATING */

@keyframes floatOrb{

    0%{
        transform:
            translate(0px,0px)
            scale(1);
    }

    25%{
        transform:
            translate(40px,-30px)
            scale(1.06);
    }

    50%{
        transform:
            translate(-30px,40px)
            scale(0.96);
    }

    75%{
        transform:
            translate(30px,20px)
            scale(1.04);
    }

    100%{
        transform:
            translate(-20px,-20px)
            scale(1);
    }
}

@keyframes floatOrb{

    0%{
        transform:
            translate3d(0px,0px,0px)
            scale(1);
    }

    25%{
        transform:
            translate3d(30px,-40px,0px)
            scale(1.08);
    }

    50%{
        transform:
            translate3d(-20px,30px,0px)
            scale(0.96);
    }

    75%{
        transform:
            translate3d(40px,20px,0px)
            scale(1.04);
    }

    100%{
        transform:
            translate3d(-30px,-30px,0px)
            scale(1);
    }
}
/* PARTICLES */

.particle{

    position:fixed;

    width:3px;
    height:3px;

    border-radius:50%;

    background:#00eaff;

    opacity:0.5;

    box-shadow:
        0 0 8px #00eaff,
        0 0 16px #00eaff;

    animation:particleFloat linear infinite;
}

.particle:nth-child(1){
    left:10%;
    animation-duration:12s;
    animation-delay:0s;
}

.particle:nth-child(2){
    left:20%;
    animation-duration:18s;
    animation-delay:2s;
}

.particle:nth-child(3){
    left:35%;
    animation-duration:14s;
    animation-delay:4s;
}

.particle:nth-child(4){
    left:50%;
    animation-duration:20s;
    animation-delay:1s;
}

.particle:nth-child(5){
    left:65%;
    animation-duration:13s;
    animation-delay:5s;
}

.particle:nth-child(6){
    left:80%;
    animation-duration:17s;
    animation-delay:3s;
}

.particle:nth-child(7){
    left:92%;
    animation-duration:15s;
    animation-delay:6s;
}

@keyframes particleFloat{

    from{
        transform:translateY(100vh);
    }

    to{
        transform:translateY(-10vh);
    }
}

    .main .block-container{
        max-width:500px;
        padding-top:4rem;
        padding-bottom:2rem;
    }

    
    /* LOGIN CARD */

    .login-card{

        background:rgba(10,18,35,0.82);

        border:1px solid rgba(0,224,255,0.16);

        border-radius:28px;

        padding:45px 40px;

        backdrop-filter:blur(14px);

        box-shadow:
            0 0 30px rgba(0,224,255,0.12),
            0 0 80px rgba(108,99,255,0.08);

        margin-bottom:20px;
    }

    /* BADGE */

    .mini-badge{

        width:max-content;

        margin:auto;
        margin-bottom:28px;

        padding:10px 20px;

        border-radius:999px;

        background:rgba(0,224,255,0.08);

        border:1px solid rgba(0,224,255,0.15);

        color:#8eeaff;

        font-size:14px;
        font-weight:700;
    }

    /* ICON */

    .lung-icon{

        text-align:center;

        font-size:82px;

        margin-bottom:10px;

        filter:drop-shadow(0 0 18px rgba(0,224,255,0.4));
    }

    /* TITLE */

    .main-title{

        text-align:center;

        color:white;

        font-size:64px;

        font-weight:900;

        margin-bottom:10px;

        text-shadow:
            0 0 20px rgba(0,224,255,0.35);
    }

    /* SUBTITLE */

    .subtitle{

        text-align:center;

        color:#b6c7e3;

        font-size:18px;

        line-height:1.7;
    }

    /* INPUTS */

    .stTextInput > div > div{

        background:rgba(255,255,255,0.06);

        border-radius:16px;

        border:1px solid rgba(255,255,255,0.08);

        transition:0.3s ease;
    }

    .stTextInput > div > div:focus-within{

        border:1px solid rgba(0,224,255,0.5);

        box-shadow:0 0 18px rgba(0,224,255,0.15);
    }

    .stTextInput input{

        color:white !important;

        font-size:16px !important;
    }

    .stTextInput label{

        color:#dce8ff !important;

        font-size:15px !important;

        font-weight:700;
    }

    /* BUTTON */

    .stButton button{

        width:100%;

        height:58px;

        border:none;

        border-radius:16px;

        background:linear-gradient(
            135deg,
            #00d4ff,
            #4f8cff,
            #6c63ff
        );

        color:white;

        font-size:18px;

        font-weight:800;

        transition:0.3s ease;

        box-shadow:
            0 0 20px rgba(0,224,255,0.3);
    }

    .stButton button:hover{

        transform:translateY(-2px);

        box-shadow:
            0 0 30px rgba(0,224,255,0.55);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# BACKGROUND GLOWS
# =========================================

st.markdown(
    """
    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    <div class="orb orb3"></div>
    <div class="orb orb4"></div>
    <div class="orb orb5"></div>

    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    """,
    unsafe_allow_html=True
)
# =========================================
# LOGIN CARD
# =========================================

components.html(
    """
    <style>

    body{
        margin:0;
        padding:0;
        background:transparent;
        font-family:Arial, sans-serif;
    }

    .hero-card{

        position:relative;

        overflow:hidden;

        border-radius:30px;

        padding:55px 40px;

        background:
            linear-gradient(
                145deg,
                rgba(15,23,42,0.95),
                rgba(2,6,23,0.98)
            );

        border:1px solid rgba(0,212,255,0.15);

        box-shadow:
            0 0 40px rgba(0,212,255,0.12),
            inset 0 0 40px rgba(255,255,255,0.02);

        text-align:center;
    }

    .hero-card::before{

        content:"";

        position:absolute;

        width:260px;
        height:260px;

        background:#00d4ff;

        border-radius:50%;

        filter:blur(120px);

        opacity:0.15;

        top:-120px;
        left:-100px;
    }

    .hero-card::after{

        content:"";

        position:absolute;

        width:220px;
        height:220px;

        background:#6c63ff;

        border-radius:50%;

        filter:blur(120px);

        opacity:0.18;

        bottom:-100px;
        right:-80px;
    }

    .badge{

        position:relative;

        z-index:2;

        display:inline-block;

        padding:10px 22px;

        border-radius:999px;

        background:rgba(0,212,255,0.08);

        border:1px solid rgba(0,212,255,0.18);

        color:#8eeaff;

        font-size:13px;

        font-weight:700;

        letter-spacing:0.5px;

        margin-bottom:28px;
    }

    .icon{

        position:relative;

        z-index:2;

        font-size:82px;

        margin-bottom:12px;

        filter:drop-shadow(0 0 22px rgba(0,212,255,0.5));
    }

    .title{

        position:relative;

        z-index:2;

        color:white;

        font-size:62px;

        font-weight:900;

        margin-bottom:12px;

        letter-spacing:1px;

        text-shadow:
            0 0 24px rgba(0,212,255,0.35);
    }

    .subtitle{

        position:relative;

        z-index:2;

        color:#b6c7e3;

        font-size:17px;

        line-height:1.7;

        max-width:500px;

        margin:auto;
    }

    .grid{

        position:absolute;

        inset:0;

        background-image:
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);

        background-size:30px 30px;

        mask-image:linear-gradient(to bottom, rgba(0,0,0,0.7), transparent);

        pointer-events:none;
    }

    </style>

    <div class="hero-card">

        <div class="grid"></div>

        <div class="badge">
            AI-Powered Clinical Intelligence
        </div>

        <div class="icon">
            🫁
        </div>

        <div class="title">
            Lung AI
        </div>

        <div class="subtitle">
            Secure access to the multimodal lung cancer analysis system
        </div>

    </div>
    """,
    height=380,
)

# =========================================
# LOGIN FORM
# =========================================

username = st.text_input("Username")

password = st.text_input(
    "Password",
    type="password"
)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Login", use_container_width=True):

    # PSEUDO LOGIN
    # PASSWORD = lungai

    if username.strip() and password == "lungai":

        st.session_state["logged_in"] = True
        st.session_state["username"] = username.strip()

        st.switch_page("pages/Home.py")

    elif not username.strip():

        st.error("Please enter a username")

    else:

        st.error("Invalid password")