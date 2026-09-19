# siddhshah-portfolio

Source for my personal portfolio: [siddhshah.netlify.app](https://siddhshah.netlify.app)

I'm a Computer Engineering student at UIUC working on GPU kernels, processor architecture, and embedded systems. The site collects the projects I've written up in depth, each with the code, the measurements, and what didn't work.

## Projects on the site

- [GPU-Accelerated CNN Inference](https://siddhshah.netlify.app/projects/project-five): fused CUDA convolution kernel using WMMA Tensor Cores
- [Out-of-Order RISC-V CPU](https://siddhshah.netlify.app/projects/project-six): RV32IM core with register renaming, gshare prediction, and a set-associative cache
- [MatShrink for Attention](https://siddhshah.netlify.app/projects/project-seven): lossless weight compression for transformer attention
- [STM32-Embedded Gesture Classifier](https://siddhshah.netlify.app/projects/project-four): TinyML gesture recognition from an accelerometer
- [FPGA-Based Ultrasonic Radar](https://siddhshah.netlify.app/projects/project-two): real-time object mapping in SystemVerilog
- [Galaxy Classification](https://siddhshah.netlify.app/projects/project-one): JAX neural network on SDSS images
- [Semantic Segmentation](https://siddhshah.netlify.app/projects/project-three): CNN vehicle segmentation on traffic-camera images

## Stack

- [Next.js](https://nextjs.org) (Pages Router) and React, statically generated
- [Tailwind CSS](https://tailwindcss.com) v4 with the typography plugin
- Content as Markdown and JSON files in `content/`, rendered with `markdown-to-jsx`
- Code blocks highlighted with Prism through `react-syntax-highlighter`
- Hosted on [Netlify](https://www.netlify.com), with the contact form handled by Netlify Forms

## Run it locally

Use Node 22, which is what Netlify builds with.

```sh
npm install
npm run dev      # http://localhost:3000
npm run build    # production build, the same command Netlify runs
```

## Where things live

```
content/
  pages/            one Markdown file per page (home, info, projects/*)
  data/config.json  header, footer, and navigation
  data/style.json   theme colors and fonts
public/
  images/           every image used by the pages
  __forms.html      static copy of the contact form (see below)
src/
  components/       layouts, sections, and form components
  css/main.css      Tailwind setup plus code block, table, and figure styles
  utils/            content loading, Markdown overrides, SEO helpers
```

## Adding a project

1. Create `content/pages/projects/<slug>.md`. The front matter needs `type: ProjectLayout`, `title`, `date`, and `description`. Add `featuredImage` (the card thumbnail) and `media` (the header image) if you have them.
2. Write the page in Markdown. Fenced code blocks get a language label and syntax highlighting when tagged with `c`, `cpp`, `cuda`, `python`, `systemverilog`, `verilog`, `javascript`, or `css`.
3. Put images in `public/images/` and reference them as `![alt text](/images/name.png "Caption shown under the figure")`. Keep parentheses out of captions, because the Markdown parser ends the caption at the first `)`.
4. The projects page lists every project newest-first by `date`. To feature one on the home page, add its path to the `projects:` list in `content/pages/index.md`.

## Contact form

The form posts to Netlify Forms. Because this is a Next.js site, Netlify can only detect a form from a static HTML file, so `public/__forms.html` holds a hidden copy of it.

- If you add, rename, or remove a field in the form (in `content/pages/*.md`), make the same change in `public/__forms.html`. Netlify only recognizes fields it saw at deploy time.
- Where the email goes is configured in the Netlify dashboard under Forms, then Submission notifications. It can't be set from this repo.
- Every submission is also stored under Forms in the dashboard.

## Deploying

Pushing to `main` triggers a Netlify build (`npm run build`, published from `.next` through `@netlify/plugin-nextjs`).

## Visual editor

For quick content edits I sometimes use Netlify's visual editor, configured in `stackbit.config.ts` and `.stackbit/`. Publishing from it commits to this repo (the commits titled "Publish"), so pull before working locally. The site builds and runs without the editor.

## Credits

Started from Netlify's [Developer Portfolio Starter](https://github.com/netlify-templates/auto-annotated-portfolio).
