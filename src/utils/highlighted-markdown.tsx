import * as React from 'react';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import c from 'react-syntax-highlighter/dist/cjs/languages/prism/c';
import cpp from 'react-syntax-highlighter/dist/cjs/languages/prism/cpp';
import css from 'react-syntax-highlighter/dist/cjs/languages/prism/css';
import js from 'react-syntax-highlighter/dist/cjs/languages/prism/javascript';
import python from 'react-syntax-highlighter/dist/cjs/languages/prism/python';
import verilog from 'react-syntax-highlighter/dist/cjs/languages/prism/verilog';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';

SyntaxHighlighter.registerLanguage('c', c);
SyntaxHighlighter.registerLanguage('cpp', cpp);
SyntaxHighlighter.registerLanguage('css', css);
SyntaxHighlighter.registerLanguage('javascript', js);
SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('verilog', verilog);

// Fence tag -> Prism grammar + label shown above the block
const LANGUAGES: Record<string, { prism: string; label: string }> = {
    c: { prism: 'c', label: 'C' },
    cpp: { prism: 'cpp', label: 'C++' },
    cuda: { prism: 'cpp', label: 'CUDA C++' },
    css: { prism: 'css', label: 'CSS' },
    javascript: { prism: 'javascript', label: 'JavaScript' },
    js: { prism: 'javascript', label: 'JavaScript' },
    python: { prism: 'python', label: 'Python' },
    systemverilog: { prism: 'verilog', label: 'SystemVerilog' },
    verilog: { prism: 'verilog', label: 'Verilog' }
};

const CodeBlock = ({ className, children }) => {
    const tag = className?.replace(/^(lang|language)-/, '') ?? '';
    const lang = LANGUAGES[tag];
    return (
        <div className="not-prose code-block">
            {lang && <div className="code-block-label">{lang.label}</div>}
            <SyntaxHighlighter
                language={lang?.prism ?? 'text'}
                style={vscDarkPlus}
                customStyle={{
                    margin: 0,
                    padding: '1.25rem 1.5rem',
                    background: 'transparent',
                    overflowX: 'auto',
                    fontFamily: 'inherit',
                    fontSize: 'inherit',
                    lineHeight: 'inherit'
                }}
                codeTagProps={{ style: { fontFamily: 'inherit', fontSize: 'inherit', lineHeight: 'inherit' } }}
            >
                {String(children).replace(/\n$/, '')}
            </SyntaxHighlighter>
        </div>
    );
};

// markdown-to-jsx uses <pre><code/></pre> for code blocks.
export default function HighlightedPreBlock({ children, ...rest }) {
    if ('type' in children && children['type'] === 'code') {
        return CodeBlock(children['props']);
    }
    return <pre {...rest}>{children}</pre>;
}

// Rendered inside the <p> markdown wraps images in, so only phrasing elements are allowed here.
export function MarkdownImage({ src, alt, title }) {
    return (
        <span className="figure">
            <a href={src} target="_blank" rel="noopener noreferrer" title="Open full size">
                <img src={src} alt={alt ?? ''} loading="lazy" />
            </a>
            {title && <span className="figcaption">{title}</span>}
        </span>
    );
}

export function MarkdownTable({ children, ...rest }) {
    return (
        <div className="table-wrap">
            <table {...rest}>{children}</table>
        </div>
    );
}
