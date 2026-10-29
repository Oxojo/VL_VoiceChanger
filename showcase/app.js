const header = document.querySelector('[data-header]');
const nav = document.querySelector('#global-nav');
const navToggle = document.querySelector('.nav-toggle');
const copyButton = document.querySelector('[data-copy]');

const updateHeader = () => header?.classList.toggle('is-scrolled', window.scrollY > 12);
updateHeader();
window.addEventListener('scroll', updateHeader, { passive: true });

const closeNav = () => {
  nav?.classList.remove('is-open');
  navToggle?.setAttribute('aria-expanded', 'false');
  document.body.style.overflow = '';
};

navToggle?.addEventListener('click', () => {
  const willOpen = navToggle.getAttribute('aria-expanded') !== 'true';
  navToggle.setAttribute('aria-expanded', String(willOpen));
  nav?.classList.toggle('is-open', willOpen);
  document.body.style.overflow = willOpen ? 'hidden' : '';
});

nav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeNav));
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeNav();
});

copyButton?.addEventListener('click', async () => {
  const commands = 'python -m vlvc setup\npython -m vlvc doctor\npython -m vlvc serve';
  const label = copyButton.querySelector('[data-copy-label]');
  try {
    await navigator.clipboard.writeText(commands);
    if (label) label.textContent = 'Copied';
  } catch {
    if (label) label.textContent = 'Select commands';
  }
  window.setTimeout(() => {
    if (label) label.textContent = 'Copy';
  }, 1800);
});