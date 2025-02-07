document.addEventListener("DOMContentLoaded", function () {
    console.log("DOMContentLoaded event fired");
  
    // --- Sidebar Toggle and Mode Switch Logic (common to all pages) ---
    const body = document.querySelector("body");
    const sidebar = body.querySelector("nav");
    const toggle = body.querySelector(".toggle");
    const searchBtn = body.querySelector(".search-box");
    const modeSwitch = body.querySelector(".toggle-switch");
    const modeText = body.querySelector(".mode-text");
  
    if (toggle) {
      toggle.addEventListener("click", () => {
        console.log("Toggle button clicked");
        sidebar.classList.toggle("close");
      });
    } else {
      console.error("Toggle element not found!");
    }
  
    if (searchBtn) {
      searchBtn.addEventListener("click", () => {
        console.log("Search button clicked");
        sidebar.classList.remove("close");
      });
    } else {
      console.error("Search button element not found!");
    }
  
    if (modeSwitch) {
      modeSwitch.addEventListener("click", () => {
        console.log("Mode switch clicked");
        body.classList.toggle("dark");
  
        if (body.classList.contains("dark")) {
          modeText.innerText = "Light mode";
        } else {
          modeText.innerText = "Dark mode";
        }
      });
    } else {
      console.error("Mode switch element not found!");
    }
  
    // --- Page-Specific Logic ---
  
    // --- Index Page (API Configuration) ---
    if (
      window.location.pathname === "/" ||
      window.location.pathname === "/index.html"
    ) {
      const modelSelect = document.getElementById("modelSelect");
      const apiKeyInput = document.getElementById("apiKeyInput");
      const saveApiConfigButton = document.getElementById("saveApiConfig");
      const configStatus = document.getElementById("configStatus");
  
      function saveApiConfiguration() {
        console.log("Saving API configuration...");
        const selectedModel = modelSelect.value;
        const apiKey = apiKeyInput.value;
  
        if (!apiKey) {
          configStatus.textContent = "Please enter an API key.";
          console.error("API key is missing!");
          return;
        }
  
        sessionStorage.setItem("selectedModel", selectedModel);
        sessionStorage.setItem("apiKey", apiKey);
        configStatus.textContent = "API configuration saved.";
        console.log(
          `API configuration saved: Model=${selectedModel}, API Key=${apiKey}`
        );
      }
  
      function loadApiConfiguration() {
        console.log("Loading API configuration...");
        const savedModel = sessionStorage.getItem("selectedModel");
        const savedApiKey = sessionStorage.getItem("apiKey");
  
        if (savedModel) {
          modelSelect.value = savedModel;
          console.log(`Loaded model from sessionStorage: ${savedModel}`);
        }
        if (savedApiKey) {
          apiKeyInput.value = savedApiKey;
          console.log(`Loaded API key from sessionStorage`);
        }
      }
  
      if (saveApiConfigButton) {
        saveApiConfigButton.addEventListener("click", saveApiConfiguration);
      } else {
        console.error("saveApiConfigButton element not found on this page!");
      }
  
      loadApiConfiguration(); // Load configuration on page load
    }
  
    // --- Data Upload Page ---
    if (window.location.pathname === "/data") {
      const fileInput = document.getElementById("fileInput");
      const uploadButton = document.getElementById("uploadButton");
      const uploadStatus = document.getElementById("uploadStatus");
  
      if (uploadButton) {
        uploadButton.addEventListener("click", function () {
          console.log("Upload button clicked");
          fileInput.click();
        });
      } else {
        console.error("uploadButton element not found on this page!");
      }
  
      if (fileInput) {
        fileInput.addEventListener("change", function () {
          console.log("File input changed");
          const file = fileInput.files[0];
          if (!file) {
            uploadStatus.textContent = "Please select a file.";
            console.error("No file selected!");
            return;
          }
  
          const formData = new FormData();
          formData.append("file", file);
  
          fetch("/upload", {
            method: "POST",
            body: formData,
          })
            .then((response) => response.json())
            .then((data) => {
              console.log("Upload response:", data);
              if (data.error) {
                uploadStatus.textContent = data.error;
              } else {
                uploadStatus.textContent = data.message;
                // Store uploadedFilename in localStorage after successful upload
                localStorage.setItem("uploadedFilename", data.filename);
              }
            })
            .catch((error) => {
              console.error("Upload error:", error);
              uploadStatus.textContent = "Error uploading file.";
            });
        });
      } else {
        console.error("fileInput element not found on this page!");
      }
    }
  
    // --- Check Data Page ---
    if (window.location.pathname === "/check") {
      const checkDataButton = document.getElementById("checkDataButton");
      const dataQualityResults = document.getElementById("dataQualityResults");
  
      if (checkDataButton) {
        checkDataButton.addEventListener("click", function () {
          console.log("Check data button clicked");
  
          // Get uploadedFilename from localStorage
          const uploadedFilename = localStorage.getItem("uploadedFilename");
  
          if (!uploadedFilename) {
            console.error("No file uploaded yet!");
            alert("Please upload a file first.");
            return;
          }
  
          fetch("/check_data", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ filename: uploadedFilename }),
          })
            .then((response) => response.json())
            .then((data) => {
              console.log("Check data response:", data);
              if (data.error) {
                dataQualityResults.textContent = data.error;
              } else {
                // Create a table element
                let resultsHTML = "<table border='1'><tr><th>Issue</th><th>Details</th></tr>";
                for (const issue in data.results) {
                  resultsHTML += `<tr><td>${issue}</td><td>${JSON.stringify(data.results[issue])}</td></tr>`;
                }
                resultsHTML += "</table>";
                dataQualityResults.innerHTML = resultsHTML;
              }
            })
            .catch((error) => {
              console.error("Error:", error);
              dataQualityResults.textContent = "Error checking data quality.";
            });
        });
      } else {
        console.error("checkDataButton element not found on this page!");
      }
    }
  
    // --- Fix Data Page ---
    if (window.location.pathname === "/fix") {
      const applyFixesButton = document.getElementById("applyFixesButton");
      const fixesResults = document.getElementById("fixesResults");
  
      if (applyFixesButton) {
        applyFixesButton.addEventListener("click", function () {
          console.log("Apply fixes button clicked");
  
          // Get uploadedFilename from localStorage
          const uploadedFilename = localStorage.getItem("uploadedFilename");
  
          if (!uploadedFilename) {
            console.error("No file uploaded yet!");
            alert("Please upload a file first.");
            return;
          }
  
          fetch("/apply_fixes", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ filename: uploadedFilename }),
          })
            .then((response) => response.json())
            .then((data) => {
              console.log("Apply fixes response:", data);
              if (data.error) {
                fixesResults.textContent = data.error;
              } else {
                // Display fixes summary in a user-friendly way
                let fixesSummary = "Fixes Applied:\n";
                const parsedFixes = JSON.parse(data.fixes_summary);
                for (const fix in parsedFixes) {
                  fixesSummary += `- ${fix}: ${parsedFixes[fix]}\n`;
                }
                fixesResults.textContent = fixesSummary;
              }
            })
            .catch((error) => {
              console.error("Error:", error);
              fixesResults.textContent = "Error applying fixes.";
            });
        });
      } else {
        console.error("applyFixesButton element not found on this page!");
      }
    }
  
    // --- Visualisation Page ---
    if (window.location.pathname === "/visualisation") {
      const generatePlotsButton = document.getElementById("generatePlotsButton");
      const plotsContainer = document.getElementById("plotsContainer");
  
      function handleGeneratePlots() {
        console.log("Generate plots button clicked");
        const plotsContainer = document.getElementById("plotsContainer");
  
        if (!plotsContainer) {
          console.error("plotsContainer element not found on this page!");
          return;
        }
        plotsContainer.innerHTML = ""; // Clear existing plots
  
        // Get uploadedFilename from localStorage
        const uploadedFilename = localStorage.getItem("uploadedFilename");
  
        if (!uploadedFilename) {
          console.error("No file uploaded yet!");
          alert("Please upload a file first.");
          return;
        }
  
        // Get the selected model and API key from sessionStorage
        const selectedModel = sessionStorage.getItem("selectedModel");
        const apiKey = sessionStorage.getItem("apiKey");
  
        if (!apiKey) {
          console.error("API key not found in sessionStorage!");
          alert("Please set the API key in the API Configuration.");
          return;
        }
  
        fetch("/generate_plots", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            filename: uploadedFilename,
            selectedModel: selectedModel,
            apiKey: apiKey,
          }),
        })
          .then((response) => {
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`); // Handle HTTP errors
            }
            return response.json();
          })
          .then((data) => {
            console.log("Generate plots response:", data);
  
            if (data.error) {
              plotsContainer.textContent = `Error: ${data.error}`; // Display error in container
              console.error("Error from server:", data.error);
            } else {
              if (!data.suggestions || !Array.isArray(data.suggestions)) {
                plotsContainer.textContent = "No suggestions received or invalid format.";
                console.warn("No suggestions or invalid format:", data);
                return; // Exit if no suggestions or wrong format
              }
            data.suggestions.forEach((plotData, index) => {
                console.log(`Processing plot ${index + 1}:`, plotData);
  
                // Skip radar chart generation
                if (plotData.suggestion.toLowerCase().includes("radar chart")) {
                    console.log(`Skipping radar chart suggestion: ${plotData.suggestion}`);
                    return; // Skip this iteration, moving to the next suggestion
                }
                 // Create plot container
                  const plotWrapper = document.createElement("div");
                  plotWrapper.classList.add("plot-container");
  
                  // Add image
                  const img = document.createElement("img");
                  if (plotData.plot) {
                    img.src = `data:image/png;base64,${plotData.plot}`;
                    img.alt = `Plot ${index + 1}`;
                    img.classList.add("plot-image");
                    console.log(`Plot ${index + 1} src:`, img.src);
  
                      img.onload = () => {
                        console.log(`Plot ${index + 1} loaded successfully.`);
                      };
  
                      img.onerror = () => {
                        console.error(`Error loading plot ${index + 1}.`);
                          img.src = ''; // set a default image or handle the error.
                          img.alt = 'Error Loading Plot';
                        };
  
                        plotWrapper.appendChild(img);
                  } else {
                    console.warn(`No plot data for suggestion ${index + 1}`);
                      plotWrapper.textContent = "No plot available for this suggestion."; // Notify the user
                  }
  
                  // Add suggestion text
                  const suggestionDiv = document.createElement("div");
                  suggestionDiv.classList.add("plot-suggestion");
                  suggestionDiv.textContent = plotData.suggestion;
                  plotWrapper.appendChild(suggestionDiv);
  
                  // Add interpretation container
                  const interpretationDiv = document.createElement("div");
                  interpretationDiv.id = `interpretation-${index}`;
                  interpretationDiv.classList.add("interpretation");
                  interpretationDiv.textContent = "Loading interpretation...";
                  plotWrapper.appendChild(interpretationDiv);
  
                  // Add chat container
                  const chatDiv = document.createElement("div");
                  chatDiv.id = `chat-${index}`;
                  chatDiv.classList.add("chat-container");
                  plotWrapper.appendChild(chatDiv);
  
                  // Add chat input
                  const chatInput = document.createElement("input");
                  chatInput.type = "text";
                  chatInput.id = `chat-input-${index}`;
                  chatInput.placeholder = "Ask a question about the graph...";
                  chatDiv.appendChild(chatInput);
  
                  // Add chat button
                  const chatButton = document.createElement("button");
                  chatButton.id = `chat-button-${index}`;
                  chatButton.textContent = "Send";
                  chatDiv.appendChild(chatButton);
  
                  plotsContainer.appendChild(plotWrapper);
  
                  // Fetch interpretation
                  fetch("/get_interpretation", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                      suggestion: plotData.suggestion,
                      filename: uploadedFilename,
                      selectedModel: selectedModel,
                      apiKey: apiKey,
                    }),
                  })
                    .then((response) => response.json())
                    .then((interpData) => {
                      console.log("Interpretation response:", interpData);
                      if (interpData.error) {
                        interpretationDiv.textContent = interpData.error;
                      } else {
                        interpretationDiv.innerHTML = `<p>${interpData.interpretation}</p>`;
                      }
                    })
                    .catch((error) => {
                      console.error("Error:", error);
                      interpretationDiv.textContent =
                        "Error loading interpretation.";
                    });
  
                  // Chat button event listener
                  chatButton.addEventListener("click", function () {
                    console.log("Chat button clicked");
                    const message = chatInput.value;
                    if (!message.trim()) return;
  
                    fetch("/graph_chat", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        message,
                        image: plotData.plot, // Send the base64 image data
                        filename: uploadedFilename,
                        selectedModel: selectedModel,
                        apiKey: apiKey,
                      }),
                    })
                      .then((response) => response.json())
                      .then((chatData) => {
                        console.log("Chat response:", chatData);
                        const responseDiv = document.createElement("div");
                        responseDiv.classList.add("model-response"); // Use a generic class
                        responseDiv.textContent = `You: ${message}\n${selectedModel}: ${chatData.response}`;
                        chatDiv.appendChild(responseDiv);
                        chatInput.value = "";
                      })
                      .catch((error) => {
                        console.error("Error:", error);
                        const errorDiv = document.createElement("div");
                        errorDiv.textContent = "Error sending message.";
                        chatDiv.appendChild(errorDiv);
                      });
                  });
  
              });
            }
          })
          .catch((error) => {
            console.error("Error:", error);
            plotsContainer.textContent = "Error generating plots.";
          });
      }
  
      if (generatePlotsButton) {
        generatePlotsButton.addEventListener("click", handleGeneratePlots);
      } else {
        console.error("generatePlotsButton element not found on this page!");
      }
    }
  });