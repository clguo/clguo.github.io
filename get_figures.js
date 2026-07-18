const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  // RSA-Net (ArXiv)
  console.log('Fetching RSA-Net...');
  await page.goto('https://arxiv.org/abs/2009.08829');
  // Usually ArXiv papers don't have images on the landing page.
  // But wait, the user's github works.
  
  // CASCNet (Springer)
  console.log('Fetching CASCNet from Springer...');
  await page.goto('https://link.springer.com/chapter/10.1007/978-981-99-8067-3_29');
  await page.waitForTimeout(3000);
  const cascnet_img = await page.$('figure img');
  if (cascnet_img) {
      await cascnet_img.screenshot({ path: 'images/thumbs/cascnet.png' });
      console.log('Saved cascnet.png');
  } else {
      console.log('CASCNet figure not found');
  }

  // AM-UNet (Springer)
  console.log('Fetching AM-UNet from Springer...');
  await page.goto('https://link.springer.com/chapter/10.1007/978-981-95-6123-0_41');
  await page.waitForTimeout(3000);
  const amunet_img = await page.$('figure img');
  if (amunet_img) {
      await amunet_img.screenshot({ path: 'images/thumbs/amunet.png' });
      console.log('Saved amunet.png');
  } else {
      console.log('AM-UNet figure not found');
  }

  await browser.close();
})();
