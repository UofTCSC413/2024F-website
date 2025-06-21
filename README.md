# Website for CSC 413 at the University of Toronto Mississauga (Fall 2024)

🔗 https://utm-csc413.github.io/2024F-website/

## Quarto Website Setup Guide

This guide will help you set up, compile, and deploy the CSC413 course website using Quarto.

### 1. Install Quarto

Visit the offical [Quarto website](https://quarto.org/docs/get-started/) to download and install Quarto. Make sure to follow the installation instructions for your operating system (Windows, macOS, Linux). You can verify the installation by running:

``` bash
quarto --version
```

### 2. Clone the Repository

Clone this repository to your local machine:

``` bash
git clone https://github.com/utm-csc413/2024F-website.git
cd 2024F-website
```

### 3. Compile the Website and Run Locally

To compile the source files (.qmd, .md) into a HTML website, run the following command after navigating to the cloned repository:

``` bash
quarto render
```

This will generate the HTML files from the source files and will be placed in the \_site/ directory by default (configurable in \_quarto.yml).

Then to preview the website locally with live reload:

``` bash
quarto preview
```

This command will start a local development server, allowing you to view the website in your browser (e.g. `http://localhost:4200`).

More details on rendering can be found in the [Quarto documentation](https://quarto.org/docs/websites/).

### 4. Deploy to GitHub Pages

To publish the compiled site directly to GitHub Pages, run the following command:

``` bash
quarto publish gh-pages
```

This command will push the contents of the `_site/` directory to the `gh-pages` branch of the repository, making it available at `https://utm-csc413.github.io/2024F-website/`.

## Colors

-   website background: #D9E3E4
-   headings: #5B888C

## Attribution

Much of the content is based on [STA 210 - Fall 2021](https://github.com/sta210-fa21/) by Dr. Maria Tackett and [STA210](https://sta210-s22.github.io/website/) by Dr. Mine Çetinkaya-Rundel.

<hr>

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" alt="Creative Commons License" style="border-width:0"/></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.