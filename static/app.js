document.addEventListener("DOMContentLoaded", () => {

    const documentsContainer =
        document.getElementById("documents");

    const selectAll =
        document.getElementById("selectAll");

    const refreshDocuments =
        document.getElementById("refreshDocuments");

    const documentFile =
        document.getElementById("documentFile");

    const uploadButton =
        document.getElementById("uploadButton");

    const uploadStatus =
        document.getElementById("uploadStatus");

    const question =
        document.getElementById("question");

    const askButton =
        document.getElementById("askButton");

    const selectionInfo =
        document.getElementById("selectionInfo");

    const answerSection =
        document.getElementById("answerSection");

    const originalQuery =
        document.getElementById("originalQuery");

    const normalizedQuery =
        document.getElementById("normalizedQuery");

    const answer =
        document.getElementById("answer");

    const sources =
        document.getElementById("sources");

    const error =
        document.getElementById("error");


    // =========================================================
    // DOCUMENTS
    // =========================================================

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

            updateSelection();

            return;
        }

        documents.forEach(doc => {

            const container =
                document.createElement("div");

            container.className =
                "document-row";


            const label =
                document.createElement("label");

            label.className =
                "document-option";


            const checkbox =
                document.createElement("input");

            checkbox.type =
                "checkbox";

            checkbox.className =
                "document-checkbox";

            checkbox.value =
                doc.id;


            const details =
                document.createElement("span");

            details.className =
                "document-details";


            const name =
                document.createElement("span");

            name.className =
                "document-name";

            name.textContent =
                doc.filename;


            const chunkCount =
                document.createElement("span");

            chunkCount.className =
                "chunk-count";

            chunkCount.textContent =
                `${doc.chunk_count} chunks`;


            details.appendChild(name);

            details.appendChild(chunkCount);


            label.appendChild(checkbox);

            label.appendChild(details);


            const deleteButton =
                document.createElement("button");

            deleteButton.type =
                "button";

            deleteButton.className =
                "delete-button";

            deleteButton.textContent =
                "Delete";


            deleteButton.addEventListener(
                "click",
                () => {

                    deleteDocument(
                        doc.id,
                        doc.filename
                    );
                }
            );


            checkbox.addEventListener(
                "change",
                () => {

                    selectAll.checked =
                        false;

                    updateSelection();
                }
            );


            container.appendChild(label);

            container.appendChild(deleteButton);

            documentsContainer.appendChild(container);
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


    // =========================================================
    // SELECT ALL
    // =========================================================

    selectAll.addEventListener(
        "change",
        () => {

            if (selectAll.checked) {

                document
                    .querySelectorAll(
                        ".document-checkbox"
                    )
                    .forEach(
                        checkbox => {
                            checkbox.checked = false;
                        }
                    );
            }

            updateSelection();
        }
    );


    // =========================================================
    // REFRESH
    // =========================================================

    refreshDocuments.addEventListener(
        "click",
        loadDocuments
    );


    // =========================================================
    // UPLOAD
    // =========================================================

    uploadButton.addEventListener(
        "click",
        uploadDocument
    );


    async function uploadDocument() {

        const file =
            documentFile.files[0];

        if (!file) {

            showUploadStatus(
                "Please select a .docx file.",
                true
            );

            return;
        }


        if (
            !file.name
                .toLowerCase()
                .endsWith(".docx")
        ) {

            showUploadStatus(
                "Only .docx files are supported.",
                true
            );

            return;
        }


        hideError();

        uploadButton.disabled = true;

        uploadButton.textContent =
            "Uploading...";


        showUploadStatus(
            "Uploading document...",
            false
        );


        try {

            const formData =
                new FormData();

            formData.append(
                "file",
                file
            );


            const response =
                await fetch(
                    "/api/documents/upload",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Document upload failed."
                );
            }


            documentFile.value = "";


            showUploadStatus(
                `${data.filename} uploaded successfully.`,
                false
            );


            await loadDocuments();

        } catch (err) {

            showUploadStatus(
                err.message,
                true
            );

        } finally {

            uploadButton.disabled = false;

            uploadButton.textContent =
                "Upload";
        }
    }


    // =========================================================
    // DELETE
    // =========================================================

    async function deleteDocument(
        documentId,
        filename
    ) {

        const confirmed =
            window.confirm(
                `Delete "${filename}"?\n\n` +
                "This will remove the document from " +
                "the index and delete the original file."
            );


        if (!confirmed) {
            return;
        }


        hideError();


        try {

            const response =
                await fetch(
                    `/api/documents/${encodeURIComponent(documentId)}`,
                    {
                        method: "DELETE"
                    }
                );


            if (!response.ok) {

                let message =
                    "Failed to delete document.";

                try {

                    const data =
                        await response.json();

                    message =
                        data.detail || message;

                } catch (_) {
                    // Response may not contain JSON.
                }


                throw new Error(message);
            }


            await loadDocuments();

        } catch (err) {

            showError(err.message);
        }
    }


    // =========================================================
    // ASK
    // =========================================================

    askButton.addEventListener(
        "click",
        askQuestion
    );


    question.addEventListener(
        "keydown",
        event => {

            if (
                event.key === "Enter" &&
                (event.metaKey || event.ctrlKey)
            ) {

                event.preventDefault();

                askQuestion();
            }
        }
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


    // =========================================================
    // RENDER ANSWER
    // =========================================================

    function renderAnswer(data) {

        /*
         * Original query exactly as entered.
         */

        originalQuery.textContent =
            data.original_query || "";


        /*
         * Normalized query used for retrieval.
         */

        normalizedQuery.textContent =
            data.normalized_query || "";


        /*
         * LLM answer.
         *
         * The model replies in Markdown (### headings, **bold**,
         * bullet lists, ...) so render it instead of showing raw
         * markup as plain text.
         */

        answer.innerHTML =
            renderMarkdown(data.answer || "");


        /*
         * Sources.
         */

        sources.innerHTML = "";


        if (
            Array.isArray(data.sources) &&
            data.sources.length > 0
        ) {

            data.sources.forEach(source => {

                const div =
                    document.createElement("div");

                div.className =
                    "source";


                const sourceName =
                    document.createElement("div");

                sourceName.className =
                    "source-name";

                sourceName.textContent =
                    source.filename;


                const sourceChunk =
                    document.createElement("div");

                sourceChunk.className =
                    "source-chunk";

                sourceChunk.textContent =
                    `Chunk: ${source.chunk_id}`;


                div.appendChild(sourceName);

                div.appendChild(sourceChunk);

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


    // =========================================================
    // ERROR
    // =========================================================

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


    // =========================================================
    // UPLOAD STATUS
    // =========================================================

    function showUploadStatus(
        message,
        isError
    ) {

        uploadStatus.textContent =
            message;

        uploadStatus.classList.remove(
            "hidden"
        );

        uploadStatus.classList.toggle(
            "upload-error",
            isError
        );

        uploadStatus.classList.toggle(
            "upload-success",
            !isError
        );
    }


    // =========================================================
    // MARKDOWN RENDERING
    //
    // The LLM answer comes back as Markdown (### headings,
    // **bold**, bullet/numbered lists, `code`). This is a small,
    // dependency-free renderer - not a full CommonMark
    // implementation, just enough to turn the model's typical
    // output into readable HTML instead of raw ###/**/- markup.
    //
    // The input is HTML-escaped first, so no raw HTML from the
    // model (or from document content it quotes) is ever inserted
    // as markup.
    // =========================================================

    function renderMarkdownInline(text) {

        let html =
            escapeHtml(text);

        // Inline code: `code`
        html = html.replace(
            /`([^`]+)`/g,
            "<code>$1</code>"
        );

        // Bold: **text** or __text__
        html = html.replace(
            /\*\*([^*]+)\*\*/g,
            "<strong>$1</strong>"
        );

        html = html.replace(
            /__([^_]+)__/g,
            "<strong>$1</strong>"
        );

        // Italic: *text* or _text_
        html = html.replace(
            /\*([^*]+)\*/g,
            "<em>$1</em>"
        );

        html = html.replace(
            /(^|[^\w])_([^_]+)_(?!\w)/g,
            "$1<em>$2</em>"
        );

        return html;
    }

    function renderMarkdown(markdown) {

        const lines =
            (markdown || "")
                .replace(/\r\n/g, "\n")
                .split("\n");

        const htmlParts = [];

        let listType = null;

        let paragraphLines = [];


        function flushParagraph() {

            if (paragraphLines.length === 0) {
                return;
            }

            htmlParts.push(
                `<p>${
                    paragraphLines
                        .map(renderMarkdownInline)
                        .join("<br>")
                }</p>`
            );

            paragraphLines = [];
        }

        function closeList() {

            if (listType) {

                htmlParts.push(
                    `</${listType}>`
                );

                listType = null;
            }
        }

        lines.forEach(rawLine => {

            const line =
                rawLine.trim();


            // Blank line -> end current paragraph/list.
            if (line === "") {

                flushParagraph();

                closeList();

                return;
            }


            // Headings: #, ##, ### ... (up to h6)
            const headingMatch =
                line.match(/^(#{1,6})\s+(.*)$/);

            if (headingMatch) {

                flushParagraph();

                closeList();

                const level =
                    headingMatch[1].length;

                htmlParts.push(
                    `<h${level}>${
                        renderMarkdownInline(headingMatch[2])
                    }</h${level}>`
                );

                return;
            }


            // Unordered list items: -, *, + followed by a space.
            const bulletMatch =
                line.match(/^[-*+]\s+(.*)$/);

            if (bulletMatch) {

                flushParagraph();

                if (listType !== "ul") {

                    closeList();

                    htmlParts.push("<ul>");

                    listType = "ul";
                }

                htmlParts.push(
                    `<li>${
                        renderMarkdownInline(bulletMatch[1])
                    }</li>`
                );

                return;
            }


            // Ordered list items: 1. text
            const numberedMatch =
                line.match(/^\d+[.)]\s+(.*)$/);

            if (numberedMatch) {

                flushParagraph();

                if (listType !== "ol") {

                    closeList();

                    htmlParts.push("<ol>");

                    listType = "ol";
                }

                htmlParts.push(
                    `<li>${
                        renderMarkdownInline(numberedMatch[1])
                    }</li>`
                );

                return;
            }


            // Plain text line -> part of the current paragraph.
            closeList();

            paragraphLines.push(line);
        });

        flushParagraph();

        closeList();


        return htmlParts.join("\n");
    }


    // =========================================================
    // HTML ESCAPE
    // =========================================================

    function escapeHtml(value) {

        const div =
            document.createElement("div");

        div.textContent =
            value;

        return div.innerHTML;
    }


    // =========================================================
    // INITIAL LOAD
    // =========================================================

    loadDocuments();

});