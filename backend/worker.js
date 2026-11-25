
/* ==========================================
   LEM v2.5
   CLOUDFLARE BACKEND
========================================== */

const CONFIG = {
    NAME: "LEM",
    VERSION: "2.5",

    DATA_URL:
        "https://raw.githubusercontent.com/BraxtonDev/CHATBOT/main/data/data.json",

    MAX_RESULTS: 3,
    MAX_MESSAGE_LENGTH: 2000
};

/* ==========================================
   CORS
========================================== */

const CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
};

/* ==========================================
   RESPONSE HELPER
========================================== */

function json(data, status = 200) {
    return new Response(JSON.stringify(data), {
        status,
        headers: {
            ...CORS_HEADERS,
            "Content-Type": "application/json",
            "Cache-Control": "no-store"
        }
    });
}

/* ==========================================
   TEXT PROCESSING
========================================== */

function tokenize(text) {
    const ignored = new Set([
        "the", "is", "a", "an", "and",
        "or", "to", "of", "in", "on",
        "for", "with", "what", "how",
        "why", "when", "where", "who",
        "does", "do", "can", "could",
        "would", "should", "i", "you",
        "it", "this", "that", "me",
        "my", "your", "about", "tell",
        "explain", "please"
    ]);

    return text
        .toLowerCase()
        .match(/[a-z0-9+#.-]+/g)
        ?.filter(word =>
            word.length > 1 && !ignored.has(word)
        ) || [];
}

/* ==========================================
   LOAD GITHUB DATA
========================================== */

async function loadKnowledge() {
    const response = await fetch(CONFIG.DATA_URL, {
        cf: {
            cacheTtl: 60,
            cacheEverything: true
        }
    });

    if (!response.ok) {
        throw new Error("Could not load GitHub data");
    }

    const data = await response.json();

    if (!Array.isArray(data.content)) {
        throw new Error("Invalid knowledge database");
    }

    return data.content.filter(
        item => typeof item === "string"
    );
}

/* ==========================================
   SEARCH KNOWLEDGE
========================================== */

function searchKnowledge(message, knowledge) {
    const words = [...new Set(tokenize(message))];

    if (!words.length) return [];

    const results = knowledge.map(content => {
        const text = content.toLowerCase();

        let score = 0;

        for (const word of words) {
            const escaped = word.replace(
                /[.*+?^${}()|[\]\\]/g,
                "\\$&"
            );

            const matches = text.match(
                new RegExp("\\b" + escaped + "\\b", "g")
            );

            if (matches) {
                score += Math.min(matches.length, 3);
            }
        }

        // Reward passages containing the complete query
        if (text.includes(message.toLowerCase().trim())) {
            score += 4;
        }

        return {
            content,
            score
        };
    });

    return results
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, CONFIG.MAX_RESULTS);
}

/* ==========================================
   GENERATE RESPONSE
========================================== */

function generateReply(message, knowledge) {
    const text = message.toLowerCase().trim();

    // Basic greetings
    if (/^(hi|hello|hey|yo|greetings)[!. ]*$/.test(text)) {
        return "Hello! I'm LEM v2.5. What would you like to learn about?";
    }

    if (/^(who are you|what are you)[?!. ]*$/.test(text)) {
        return "I'm LEM v2.5, a lightweight chatbot that searches a connected knowledge database to answer questions.";
    }

    const results = searchKnowledge(message, knowledge);

    if (!results.length) {
        return "I couldn't find relevant information in my current knowledge database. My knowledge can be expanded by adding more passages to data.json.";
    }

    const passages = results.map(
        (item, index) =>
            `${index + 1}. ${item.content}`
    );

    return "Here's what I found in my knowledge database:\n\n" +
        passages.join("\n\n");
}

/* ==========================================
   MAIN WORKER
========================================== */

export default {
    async fetch(request, env, ctx) {

        // Handle browser preflight
        if (request.method === "OPTIONS") {
            return new Response(null, {
                status: 204,
                headers: CORS_HEADERS
            });
        }

        const url = new URL(request.url);

        // Status endpoint
        if (
            url.pathname === "/api/status" &&
            request.method === "GET"
        ) {
            return json({
                name: CONFIG.NAME,
                version: CONFIG.VERSION,
                status: "online",
                backend: "connected"
            });
        }

        // Chat endpoint
        if (
            url.pathname === "/api/chat" &&
            request.method === "POST"
        ) {
            try {
                const body = await request.json();

                if (
                    typeof body.message !== "string" ||
                    !body.message.trim()
                ) {
                    return json({
                        error: "A message is required."
                    }, 400);
                }

                const message = body.message.trim();

                if (
                    message.length > CONFIG.MAX_MESSAGE_LENGTH
                ) {
                    return json({
                        error: "Message is too long."
                    }, 413);
                }

                // Load knowledge
                const knowledge = await loadKnowledge();

                // Generate response
                const reply = generateReply(
                    message,
                    knowledge
                );

                return json({
                    reply,
                    model: CONFIG.NAME,
                    version: CONFIG.VERSION,
                    passages_loaded: knowledge.length
                });

            } catch (error) {
                console.error("LEM ERROR:", error);

                return json({
                    error: "LEM encountered a backend error.",
                    reply: "I couldn't process your message right now. Please try again."
                }, 500);
            }
        }

        return json({
            error: "Endpoint not found.",
            available: [
                "/api/status",
                "/api/chat"
            ]
        }, 404);
    }
};
