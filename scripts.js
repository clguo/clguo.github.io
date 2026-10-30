// Global variables
let allPublications = [];
let showingSelected = true;

// Initialize the page
document.addEventListener('DOMContentLoaded', function () {
  // Load publications data
  loadPublications();

  // Initialize animation delays for sections
  const sections = document.querySelectorAll('section');
  sections.forEach((section, index) => {
    section.style.animationDelay = `${index * 0.1}s`;
  });

  // Add event listener for toggle button
  const toggleButton = document.getElementById('toggle-publications');
  if (toggleButton) {
    toggleButton.addEventListener('click', togglePublications);
  }
});

// Load publications from global variable (avoiding CORS issues)
function loadPublications() {
  if (typeof publicationsData !== 'undefined') {
    allPublications = publicationsData.publications;
    renderPublications(true);
  } else {
    displayFallbackPublications();
  }
}

// Fallback if data is missing
function displayFallbackPublications() {
  const container = document.getElementById('publications-container');
  container.innerHTML = `Unable to load publications data.`;
}

// Toggle between showing all or selected publications
function togglePublications() {
  showingSelected = !showingSelected;
  renderPublications(showingSelected);

  // Update button text
  const toggleButton = document.getElementById('toggle-publications');
  toggleButton.textContent = showingSelected ? 'Show All' : 'Show Selected';
  const toggleHeader = document.getElementById('toggle-header');
  toggleHeader.textContent = showingSelected ? 'Selected Publications' : 'All Publications';
}

// Render publications based on selection state
function renderPublications(selectedOnly) {
  const publicationsContainer = document.getElementById('publications-container');
  publicationsContainer.innerHTML = '';

  const pubsToShow = selectedOnly ?
    allPublications.filter(pub => pub.selected === 1) :
    allPublications;

  pubsToShow.forEach(publication => {
    const pubElement = createPublicationElement(publication);
    publicationsContainer.appendChild(pubElement);
  });
}

// Create HTML element for a publication
function createPublicationElement(publication) {
  const pubItem = document.createElement('div');
  pubItem.className = 'publication-item';

  // Create thumbnail
  const thumbnail = document.createElement('div');
  thumbnail.className = 'pub-thumbnail';
  const showTextCover = () => {
    thumbnail.classList.add('pub-label');
    thumbnail.onclick = null;
    thumbnail.textContent = publication.label || publication.title.split(':')[0];
    thumbnail.setAttribute('aria-hidden', 'true');
  };

  if (publication.thumbnail) {
    thumbnail.onclick = () => openModal(publication.thumbnail, publication.title);
    const thumbnailImg = document.createElement('img');
    thumbnailImg.onerror = showTextCover;
    thumbnailImg.src = publication.thumbnail;
    thumbnailImg.alt = `${publication.title} thumbnail`;
    thumbnail.appendChild(thumbnailImg);
  } else {
    showTextCover();
  }

  // Create content container
  const content = document.createElement('div');
  content.className = 'pub-content';

  // Add title
  const title = document.createElement('div');
  title.className = 'pub-title';
  title.textContent = publication.title;
  content.appendChild(title);

  // Add authors with highlight
  const authors = document.createElement('div');
  authors.className = 'pub-authors';

  // Format authors with highlighting
  let authorsHTML = '';
  publication.authors.forEach((author, index) => {
    if (author.includes('Changlu Guo')) {
      authorsHTML += `<span class="highlight-name">${author}</span>`;
    } else {
      authorsHTML += author;
    }

    if (index < publication.authors.length - 1) {
      authorsHTML += ', ';
    }
  });

  authors.innerHTML = authorsHTML;
  content.appendChild(authors);

  // Add venue with award if present
  const venueContainer = document.createElement('div');
  venueContainer.className = 'pub-venue-container';

  const venue = document.createElement('div');
  venue.className = 'pub-venue';
  venue.textContent = publication.venue;
  venueContainer.appendChild(venue);

  // Add award if it exists
  if (publication.award && publication.award.length > 0) {
    const award = document.createElement('div');
    award.className = 'pub-award';
    award.textContent = publication.award;
    venueContainer.appendChild(award);
  }

  content.appendChild(venueContainer);

  // Add links if they exist
  if (publication.links) {
    const links = document.createElement('div');
    links.className = 'pub-links';

    if (publication.links.pdf) {
      const pdfLink = document.createElement('a');
      pdfLink.href = publication.links.pdf;
      pdfLink.textContent = '[PDF]';
      links.appendChild(pdfLink);
    }

    if (publication.links.code) {
      const codeLink = document.createElement('a');
      codeLink.href = publication.links.code;
      codeLink.textContent = '[Code]';
      links.appendChild(codeLink);
    }

    if (publication.links.project) {
      const projectLink = document.createElement('a');
      projectLink.href = publication.links.project;
      projectLink.textContent = '[Project Page]';
      links.appendChild(projectLink);
    }

    content.appendChild(links);
  }

  // Assemble the publication item
  pubItem.appendChild(thumbnail);
  pubItem.appendChild(content);

  return pubItem;
}

// Modal functionality for viewing original images
let modalCloseTimer;
let modalPreviousFocus;

function openModal(imageSrc, imageTitle = 'Publication figure') {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  const closeButton = modal.querySelector('.modal-close');
  clearTimeout(modalCloseTimer);
  modalPreviousFocus = document.activeElement;
  modal.style.display = "block";
  modal.setAttribute('aria-hidden', 'false');
  document.body.classList.add('modal-open');
  setTimeout(() => {
    modal.classList.add('show');
  }, 10);
  modalImg.src = imageSrc;
  modalImg.alt = imageTitle;
  closeButton.focus();
}

function closeModal() {
  const modal = document.getElementById('imageModal');
  if (modal.getAttribute('aria-hidden') === 'true') return;
  modal.classList.remove('show');
  modal.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('modal-open');
  clearTimeout(modalCloseTimer);
  modalCloseTimer = setTimeout(() => {
    modal.style.display = "none";
  }, 300);
  if (modalPreviousFocus && typeof modalPreviousFocus.focus === 'function') {
    modalPreviousFocus.focus();
  }
}

// Close modal when clicking outside the image
window.onclick = function (event) {
  const modal = document.getElementById('imageModal');
  if (event.target == modal) {
    closeModal();
  }
}

// Close an enlarged figure with the Escape key.
document.addEventListener('keydown', (event) => {
  const modal = document.getElementById('imageModal');
  if (event.key === 'Escape' && modal.getAttribute('aria-hidden') === 'false') {
    event.preventDefault();
    closeModal();
  }
});
