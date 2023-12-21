/* ====================================================== Nav ====================================================== */
const navMenu = document.getElementById('nav-menu'),
      navToggle = document.getElementById('nav-toggle'),
      navClose = document.getElementById('nav-close')

if(navToggle){
    navToggle.addEventListener('click', () =>{
        navMenu.classList.add('show-menutab')
    })
}

if(navClose){
    navClose.addEventListener('click', () =>{
        navMenu.classList.remove('show-menutab')
    })
}

const navLink = document.querySelectorAll('.nav_link')

function linkAction(){
    const navMenu = document.getElementById('.nav-menu')
    navMenu.classList.remove('.show-menutab')
}
navLink.forEach(n => n.addEventListener('click', linkAction))

/* ====================================================== Scrolled ====================================================== */

const navbanner = document.querySelector('.header');

window.addEventListener('scroll', () =>{
    if (window.scrollY > 200){
        navbanner.classList.add('header_scrolled');
    }else{
        navbanner.classList.remove('header_scrolled');
    }
}) 