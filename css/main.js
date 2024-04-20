// Utility function to get parent element with specified selector
const getParentElement = (element, selector) => {
  let currentElement = element;
  while (currentElement && currentElement.nodeType === 1) {
    if (currentElement.matches(selector)) {
      return currentElement;
    }
    currentElement = currentElement.parentNode;
  }
  return null;
};

// Utility function to toggle class for an element
const toggleClass = (element, className) => {
  if (element.classList) {
    element.classList.toggle(className);
  } else {
    const classes = element.className.split(' ');
    const index = classes.indexOf(className);
    if (index > -1) {
      classes.splice(index, 1);
    } else {
      classes.push(className);
    }
    element.className = classes.join(' ');
  }
};

// Get references to the elements
const main = document.getElementById('main');
const mainParent = getParentElement(main, '#content > .container');
const extensions = document.getElementById('extensions');

// Add an event listener to the main parent element
mainParent.addEventListener('click', () => {
  // Toggle the visibility of the extensions
  toggleClass(extensions, 'visible');

  // Add/remove padding for the main parent element
  toggleClass(mainParent, 'no-padding');
});

// Add initial styles
main.classList.add('header_bar');
mainParent.style.padding = '0';
mainParent.style.margin = '0';
const grandParent = getParentElement(mainParent, 'body > .container');
if (grandParent) {
  grandParent.style.padding = '0';
}
