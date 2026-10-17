// Global variables
let allPublications = [];
let showingSelected = true;
let scholarStats = null;

// Initialize the page
document.addEventListener('DOMContentLoaded', function () {
  // Load publications data
  loadPublications();
  loadScholarStats();

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

function citationKey(title) {
  return title.toLowerCase().replace(/[^a-z0-9]/g, '');
}

// Serve the last successful daily snapshot; never scrape Scholar in a visitor's browser.
async function loadScholarStats() {
  try {
    const response = await fetch('data/scholar.json', {
      cache: 'no-store',
      signal: AbortSignal.timeout(8000)
    });
    if (!response.ok) throw new Error('Citation data unavailable');
    const data = await response.json();
    const keys = ['citations', 'h_index', 'i10_index'];
    const updated = new Date(data.updated_at);
    if (data.profile_id !== 'y6XNTKEAAAAJ' ||
        !keys.every(key => Number.isSafeInteger(data[key]) && data[key] >= 0) ||
        Number.isNaN(updated.getTime()) || !data.publications ||
        typeof data.publications !== 'object' || Array.isArray(data.publications) ||
        !Object.values(data.publications).every(value => Number.isSafeInteger(value) && value >= 0)) {
      throw new Error('Invalid citation data');
    }
    scholarStats = data;
    document.getElementById('stat-citations').textContent = data.citations.toLocaleString('en-US');
    document.getElementById('stat-h-index').textContent = data.h_index;
    document.getElementById('stat-i10-index').textContent = data.i10_index;
    const date = updated.toLocaleDateString('en-GB', {
      day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC'
    });
    const note = document.getElementById('scholar-updated');
    note.textContent = `Google Scholar · Last updated ${date}`;
    note.title = `Last successful refresh: ${data.updated_at}`;
    renderPublications(showingSelected);
  } catch (error) {
    // Preserve the dated HTML fallback if the snapshot cannot be loaded.
    console.warn('Using saved citation figures:', error.message);
  }
}

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
    thumbnail.onclick = () => openModal(publication.thumbnail);
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
    const citations = scholarStats?.publications[citationKey(publication.title)];
    if (/Citations/i.test(publication.award) && Number.isSafeInteger(citations)) {
      award.textContent = `${citations.toLocaleString('en-US')} Citations`;
      award.title = `Google Scholar · Updated ${scholarStats.updated_at.slice(0, 10)}`;
    } else {
      award.textContent = publication.award;
    }
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
function openModal(imageSrc) {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  modal.style.display = "block";
  setTimeout(() => {
    modal.classList.add('show');
  }, 10);
  modalImg.src = imageSrc;
}

function closeModal() {
  const modal = document.getElementById('imageModal');
  modal.classList.remove('show');
  setTimeout(() => {
    modal.style.display = "none";
  }, 300);
}

// Close modal when clicking outside the image
window.onclick = function (event) {
  const modal = document.getElementById('imageModal');
  if (event.target == modal) {
    closeModal();
  }
}
