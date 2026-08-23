const documentsContainer =
    document.getElementById("documents");

const selectAll =
    document.getElementById("selectAll");

const refreshDocuments =
    document.getElementById("refreshDocuments");

const question =
    document.getElementById("question");

const askButton =
    document.getElementById("askButton");

const selectionInfo =
    document.getElementById("selectionInfo");

const answerSection =
    document.getElementById("answerSection");

const answer =
    document.getElementById("answer");

const sources =
    document.getElementById("sources");

const error =
    document.getElementById("error");


async function loadDocuments() {

    documentsContainer.innerHTML =
        '<p class="loading">Loading documents...</p>';

    try {

        const response =
            await fetch("/api/documents/");

        if (!response.ok) {
            throw new Error(
                "Failed to load documents."
            );
        }

        const documents =
            await response.json();

        renderDocuments(documents);

    } catch (err) {

        documentsContainer.innerHTML =
            `<p class="error">${escapeHtml(err.message)}</p>`;
    }
}


function renderDocuments(documents) {

    documentsContainer.innerHTML = "";

    if (documents.length === 0) {

        documentsContainer.innerHTML =
            '<p class="loading">No documents available.</p>';

        return;
    }

    documents.forEach(doc => {

        const label =
            document.createElement("label");

        label.className =
            "document-option";

        label.innerHTML = `
            <input
                type="checkbox"
                class="document-checkbox"
                value="${escapeHtml(doc.id)}"
            >

            <span>
                <span class="document-name">
                    ${escapeHtml(doc.filename)}
                </span>

                <span class="chunk-count">
                    ${doc.chunk_count} chunks
                </span>
            </span>
        `;

        documentsContainer.appendChild(label);
    });

    document
        .querySelectorAll(".document-checkbox")
        .forEach(checkbox => {

            checkbox.addEventListener(
                "change",
                updateSelection
            );
        });

    updateSelection();
}


function getSelectedDocumentIds() {

    return Array.from(
        document.querySelectorAll(
            ".document-checkbox:checked"
        )
    ).map(
        checkbox => checkbox.value
    );
}


function updateSelection() {

    const selected =
        getSelectedDocumentIds();

    if (selectAll.checked) {

        selectionInfo.textContent =
            "Search all documents";

        return;
    }

    if (selected.length === 0) {

        selectionInfo.textContent =
            "No document selected";

        return;
    }

    selectionInfo.textContent =
        `${selected.length} document${
            selected.length === 1 ? "" : "s"
        } selected`;
}


selectAll.addEventListener(
    "change",
    () => {

        const checkboxes =
            document.querySelectorAll(
                ".document-checkbox"
            );

        checkboxes.forEach(
            checkbox => {
                checkbox.checked = false;
            }
        );

        updateSelection();
    }
);


document.addEventListener(
    "change",
    event => {

        if (
            event.target.classList.contains(
                "document-checkbox"
            )
        ) {
            selectAll.checked = false;
            updateSelection();
        }
    }
);


refreshDocuments.addEventListener(
    "click",
    loadDocuments
);


askButton.addEventListener(
    "click",
    askQuestion
);


async function askQuestion() {

    const query =
        question.value.trim();

    if (!query) {

        showError(
            "Please enter a question."
        );

        return;
    }

    const selected =
        getSelectedDocumentIds();

    if (
        !selectAll.checked &&
        selected.length === 0
    ) {

        showError(
            "Select at least one document or choose Search all documents."
        );

        return;
    }

    hideError();

    askButton.disabled = true;

    askButton.textContent =
        "Asking...";

    answerSection.classList.add(
        "hidden"
    );

    try {

        const payload = {
            question: query,
            document_ids:
                selectAll.checked
                    ? null
                    : selected
        };

        const response =
            await fetch(
                "/api/chat",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(payload)
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Request failed."
            );
        }

        renderAnswer(data);

    } catch (err) {

        showError(err.message);

    } finally {

        askButton.disabled = false;

        askButton.textContent =
            "Ask";
    }
}


function renderAnswer(data) {

    answer.textContent =
        data.answer || "";

    sources.innerHTML = "";

    if (
        data.sources &&
        data.sources.length > 0
    ) {

        data.sources.forEach(source => {

            const div =
                document.createElement("div");

            div.className =
                "source";

            div.innerHTML = `
                <div class="source-name">
                    ${escapeHtml(source.filename)}
                </div>

                <div class="source-chunk">
                    Chunk:
                    ${escapeHtml(source.chunk_id)}
                </div>
            `;

            sources.appendChild(div);
        });

    } else {

        sources.innerHTML =
            '<p class="loading">No sources returned.</p>';
    }

    answerSection.classList.remove(
        "hidden"
    );
}


function showError(message) {

    error.textContent =
        message;

    error.classList.remove(
        "hidden"
    );
}


function hideError() {

    error.classList.add(
        "hidden"
    );
}


function escapeHtml(value) {

    const div =
        document.createElement("div");

    div.textContent =
        String(value);

    return div.innerHTML;
}


loadDocuments();