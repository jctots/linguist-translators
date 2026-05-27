/**
 * Ollama — run large language models locally or on a private server
 * Homepage: https://ollama.com/
 * API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
 *
 * Setup:
 *   1. Install Ollama: https://ollama.com/download
 *   2. Pull a model: ollama pull translategemma (recommended) or llama3.2
 *   3. Set serverUrl below (default: http://localhost:11434)
 *   4. Set model to the name of the model you pulled
 *
 * For a private server, set serverUrl to e.g. https://ollama.yourhomelab.com
 * If your server requires an API key, set apiKey accordingly.
 *
 * Recommended model: translategemma (https://ollama.com/library/translategemma)
 * A translation-specific model based on Gemma. Works well with the prompt
 * format used by this translator.
 */
class OllamaTranslator {
	// URL of your Ollama instance
	// Local:   http://localhost:11434
	// Private: https://ollama.yourhomelab.com
	serverUrl = 'http://localhost:11434';

	// Model to use for translation (must be pulled on your Ollama instance)
	// Recommended: translategemma
	// Others:      llama3.2, gemma3, mistral, qwen2.5
	model = 'translategemma:latest';

	// API key — leave empty if your server does not require one
	apiKey = '';

	// Per-request timeout in milliseconds (default: 120s — LLM inference can be slow)
	// Increase for large pages or slow hardware; decrease for fast local servers.
	inferenceTimeout = 120_000;

	// Send a single prompt to /api/chat and return the model's text response.
	// Timeout via Promise.race — AbortSignal cannot cross the postMessage bridge
	// used by Linguist's custom translator sandbox on Chromium MV3.
	_chat = async (prompt) => {
		const base = this.serverUrl.replace(/\/+$/, '');
		const headers = { 'Content-Type': 'application/json' };
		if (this.apiKey) headers['Authorization'] = `Bearer ${this.apiKey}`;

		const fetchPromise = fetch(`${base}/api/chat`, {
			method: 'POST',
			headers,
			body: JSON.stringify({
				model: this.model,
				messages: [{ role: 'user', content: prompt }],
				stream: false,
			}),
		});
		let timeoutId;
		const timeoutPromise = new Promise((_, reject) => {
			timeoutId = setTimeout(
				() => reject(new Error(`Ollama request timed out after ${this.inferenceTimeout / 1000}s`)),
				this.inferenceTimeout,
			);
		});

		const response = await Promise.race([fetchPromise, timeoutPromise]);
		clearTimeout(timeoutId);

		if (!response.ok) {
			throw new Error(`Ollama error ${response.status}: ${response.statusText}`);
		}

		const data = await response.json();
		return data.message?.content?.trim() ?? '';
	};

	translate = async (text, from, to) => {
		const [result] = await this.translateBatch([text], from, to);
		return result;
	};

	translateBatch = async (texts, from, to) => {
		const results = new Array(texts.length);

		// Group texts into batches that fit within the length limit,
		// then translate each batch as a single Ollama request.
		const batches = [];
		let currentBatch = [];
		let currentLength = 0;

		for (let i = 0; i < texts.length; i++) {
			const text = texts[i];
			if (currentBatch.length > 0 && currentLength + text.length > this.getLengthLimit()) {
				batches.push(currentBatch);
				currentBatch = [];
				currentLength = 0;
			}
			currentBatch.push({ index: i, text });
			currentLength += text.length;
		}
		if (currentBatch.length > 0) batches.push(currentBatch);

		for (const batch of batches) {
			const toName = OllamaTranslator.langName(to);
			const toCode = to;
			// JSON encoding avoids any collision between source text content and prompt structure.
			const jsonInput = JSON.stringify(batch.map(({ text }) => text));

			let prompt;
			if (from && from !== 'auto') {
				const fromName = OllamaTranslator.langName(from);
				const fromCode = from;
				prompt =
					`You are a professional ${fromName} (${fromCode}) to ${toName} (${toCode}) translator. ` +
					`Translate each string in the JSON array below to ${toName}. ` +
					`Return only a JSON array of translated strings in the same order, with no other text.\n\n` +
					jsonInput;
			} else {
				prompt =
					`You are a professional translator. ` +
					`Translate each string in the JSON array below to ${toName} (${toCode}). ` +
					`Return only a JSON array of translated strings in the same order, with no other text.\n\n` +
					jsonInput;
			}

			const rawOutput = await this._chat(prompt);

			// Extract and parse the JSON array from the model response.
			// Fallback to original text per item if the model output is not valid JSON.
			let translations = null;
			try {
				const match = rawOutput.match(/\[[\s\S]*\]/);
				translations = JSON.parse(match ? match[0] : rawOutput);
			} catch {
				// intentionally empty — translations stays null, fallback applied below
			}

			for (let i = 0; i < batch.length; i++) {
				results[batch[i].index] =
					Array.isArray(translations) && typeof translations[i] === 'string'
						? translations[i]
						: batch[i].text;
			}
		}

		return results;
	};

	getLengthLimit = () => 8000;
	getRequestsTimeout = () => 500;
	checkLimitExceeding = (text) => {
		const textLength = !Array.isArray(text)
			? text.length
			: text.reduce((len, t) => len + t.length, 0);
		return textLength - this.getLengthLimit();
	};

	static isSupportedAutoFrom = () => true;
	// prettier-ignore
	static getSupportedLanguages = () => [
		"en", "nl", "de", "fr", "es", "it", "pt", "ru",
		"ja", "zh", "ko", "ar", "tr", "pl", "sv", "da",
		"fi", "nb", "cs", "sk", "hu", "ro", "bg", "uk",
		"el", "he", "hi", "th", "vi", "id", "ms"
	];

	static _dn = new Intl.DisplayNames(['en'], { type: 'language' });
	static langName = (code) => OllamaTranslator._dn.of(code) ?? code;
}

OllamaTranslator;
