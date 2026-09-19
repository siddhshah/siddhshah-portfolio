import classNames from 'classnames';
import * as React from 'react';

import { Annotated } from '@/components/Annotated';
import { DynamicComponent } from '@/components/components-registry';
import { mapStylesToClassNames as mapStyles } from '@/utils/map-styles-to-class-names';

// Netlify only detects forms in static files, so submissions are POSTed to a static copy of the form.
const FORMS_ENDPOINT = '/__forms.html';
const FALLBACK_EMAIL = 'siddh.shah90@gmail.com';
const NOTIFICATION_SUBJECT = 'New message from your portfolio site';

type Status = 'idle' | 'sending' | 'sent' | 'error';

export default function FormBlock(props) {
    const { elementId, className, fields = [], submitLabel, styles = {} } = props;
    const [status, setStatus] = React.useState<Status>('idle');
    const textAlign = styles.self?.textAlign ?? 'left';

    if (fields.length === 0) {
        return null;
    }

    async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault();
        const form = event.currentTarget;
        setStatus('sending');
        try {
            const body = new URLSearchParams(new FormData(form) as unknown as Record<string, string>).toString();
            const response = await fetch(FORMS_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body
            });
            if (!response.ok) {
                throw new Error(`Form submission failed with status ${response.status}`);
            }
            form.reset();
            setStatus('sent');
        } catch {
            setStatus('error');
        }
    }

    return (
        <Annotated content={props}>
            <form className={className} name={elementId} id={elementId} onSubmit={handleSubmit}>
                <div className="grid gap-6 sm:grid-cols-2">
                    <input type="hidden" name="form-name" value={elementId} />
                    <input type="hidden" name="subject" value={NOTIFICATION_SUBJECT} />
                    <div hidden aria-hidden="true">
                        <label>
                            Leave this field empty
                            <input name="bot-field" tabIndex={-1} autoComplete="off" />
                        </label>
                    </div>
                    {fields.map((field, index) => {
                        return <DynamicComponent key={index} {...field} />;
                    })}
                </div>
                <div className={classNames('mt-8', mapStyles({ textAlign }))}>
                    <button
                        type="submit"
                        disabled={status === 'sending'}
                        className="inline-flex items-center justify-center px-5 py-4 text-lg transition border-2 border-current hover:bottom-shadow-6 hover:-translate-y-1.5 disabled:opacity-50 disabled:pointer-events-none"
                    >
                        {status === 'sending' ? 'Sending…' : submitLabel}
                    </button>
                    <div className="mt-6 text-lg" aria-live="polite">
                        {status === 'sent' && <p>Thanks! Your message has been sent and I&apos;ll get back to you soon.</p>}
                        {status === 'error' && (
                            <p role="alert">
                                Sorry, that didn&apos;t go through. Please email me directly at{' '}
                                <a className="underline" href={`mailto:${FALLBACK_EMAIL}`}>
                                    {FALLBACK_EMAIL}
                                </a>
                                .
                            </p>
                        )}
                    </div>
                </div>
            </form>
        </Annotated>
    );
}
