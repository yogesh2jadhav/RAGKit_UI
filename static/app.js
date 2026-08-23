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

    const answer =
        document.getElementById("answer");

    const sources =
        document.getElementById("sources");

    const error =
        document.getElementById("error");


    /*
     * Load indexed documents.
     */
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


    /*
     * Render documents and their delete buttons.
     */
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

            documentsContainer.appendChild(
                container
            );
        });

        updateSelection();
    }


    /*
     * Return selected document IDs.
     */
    function getSelectedDocumentIds() {

        return Array.from(
            document.querySelectorAll(
                ".document-checkbox:checked"
            )
        ).map(
            checkbox => checkbox.value
        );
    }


    /*
     * Update the selection message.
     */
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


    /*
     * Search all documents.
     */
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


    /*
     * Refresh document list.
     */
    refreshDocuments.addEventListener(
        "click",
        loadDocuments
    );


    /*
     * Upload button.
     */
    uploadButton.addEventListener(
        "click",
        uploadDocument
    );


    /*
     * Ask button.
     */
    askButton.addEventListener(
        "click",
        askQuestion
    );


    /*
     * Allow Cmd/Ctrl + Enter to submit a question.
     */
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


    /*
     * Upload a document.
     */
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


    /*
     * Delete a document.
     */
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


    /*
     * Ask the RAG question.
     */
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


    /*
     * Render RAG answer and sources.
     */
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

                div.appendChild(
                    sourceName
                );

                div.appendChild(
                    sourceChunk
                );

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


    /*
     * Show an application error.
     */
    function showError(message) {

        error.textContent =
            message;

        error.classList.remove(
            "hidden"
        );
    }


    /*
     * Hide an application error.
     */
    function hideError() {

        error.classList.add(
            "hidden"
        );
    }


    /*
     * Show upload status.
     */
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


    /*
     * Initial document load.
     */
    loadDocuments();
});