document.addEventListener('DOMContentLoaded', () => {
    const meanFeaturesDiv = document.getElementById('mean-features');
    const seFeaturesDiv = document.getElementById('se-features');
    const worstFeaturesDiv = document.getElementById('worst-features');
    
    // Feature definition matching the dataset order exactly
    const featureNames = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", 
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", 
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", 
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ];

    // Some sample data for Malignant (from row 1)
    const sampleMalignant = [
        17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
        1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
        25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
    ];

    // Some sample data for Benign (from row 21)
    const sampleBenign = [
        13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,
        0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,
        15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259
    ];

    // Generate input fields dynamically
    function createInput(name, id, index) {
        const wrapper = document.createElement('div');
        wrapper.className = 'input-wrapper';
        // Add cascading animation delay
        wrapper.style.animationDelay = `${0.3 + (index * 0.03)}s`;
        
        const label = document.createElement('label');
        label.setAttribute('for', id);
        // Clean up the label name
        label.textContent = name.replace(/_/g, ' ');
        
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.id = id;
        input.name = id;
        input.required = true;
        input.placeholder = "0.0";
        
        wrapper.appendChild(label);
        wrapper.appendChild(input);
        return wrapper;
    }

    featureNames.forEach((name, index) => {
        const inputElement = createInput(name, `feature_${index}`, index);
        
        if (index < 10) {
            meanFeaturesDiv.appendChild(inputElement);
        } else if (index < 20) {
            seFeaturesDiv.appendChild(inputElement);
        } else {
            worstFeaturesDiv.appendChild(inputElement);
        }
    });

    // Auto-Fill Functionality
    let toggleMalignant = true;
    document.getElementById('autoFillBtn').addEventListener('click', (e) => {
        e.preventDefault();
        const dataToUse = toggleMalignant ? sampleMalignant : sampleBenign;
        
        // Add a nice visual effect to the button
        const btn = e.currentTarget;
        const icon = btn.querySelector('i');
        icon.classList.add('fa-spin');
        
        featureNames.forEach((_, index) => {
            const input = document.getElementById(`feature_${index}`);
            // Add a tiny delay for a cascading fill effect
            setTimeout(() => {
                input.value = dataToUse[index];
                // Trigger input event to style it if needed
                input.dispatchEvent(new Event('input'));
            }, index * 10);
        });

        setTimeout(() => {
            icon.classList.remove('fa-spin');
        }, 300);

        toggleMalignant = !toggleMalignant;
    });

    // Form Submission
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Updates for loading state
        const submitBtn = document.getElementById('predictBtn');
        const contentSpan = submitBtn.querySelector('.btn-content');
        const loaderSpan = submitBtn.querySelector('.btn-loader');
        
        submitBtn.disabled = true;
        contentSpan.style.opacity = '0.5';
        loaderSpan.style.display = 'inline-block';
        
        const startTime = performance.now();

        try {
            // Collect features in correct order
            const features = [];
            for (let i = 0; i < 30; i++) {
                const val = parseFloat(document.getElementById(`feature_${i}`).value);
                features.push(val);
            }

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ features: features })
            });

            if (!response.ok) throw new Error('Prediction request failed');
            
            const result = await response.json();
            const latency = Math.round(performance.now() - startTime);
            
            displayResult(result, latency);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the prediction engine. Is the backend running?');
        } finally {
            submitBtn.disabled = false;
            contentSpan.style.opacity = '1';
            loaderSpan.style.display = 'none';
        }
    });

    function displayResult(data, latency) {
        const resultCard = document.getElementById('resultCard');
        const resultContent = document.getElementById('resultContent');
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceValue = document.getElementById('confidenceValue');
        const latencyVal = document.getElementById('latencyVal');

        // Reset classes
        resultCard.classList.remove('malignant', 'benign');
        
        // Apply new class
        const isMalignant = data.prediction === 'Malignant';
        resultCard.classList.add(isMalignant ? 'malignant' : 'benign');

        // Update confidence
        const probPct = (data.probability * 100).toFixed(2);
        confidenceBar.style.width = `${probPct}%`;
        confidenceValue.textContent = `${probPct}%`;
        
        // Update latency
        latencyVal.textContent = `${latency} ms`;

        // Update main content
        const iconClass = isMalignant ? 'fa-virus-covid' : 'fa-shield-heart';
        const badgeClass = isMalignant ? 'malignant-badge' : 'benign-badge';
        
        resultContent.innerHTML = `
            <div class="prediction-badge ${badgeClass}">
                <i class="fa-solid ${iconClass}"></i>
                ${data.prediction}
            </div>
            <div class="prob-display">
                Neural Confidence: <span>${probPct}%</span>
            </div>
            <p style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.9rem;">
                ${isMalignant 
                    ? 'High-risk cell characteristics detected. Immediate clinical review recommended.' 
                    : 'Cell characteristics appear stable. Routine monitoring advised.'}
            </p>
        `;
        resultContent.classList.remove('empty-state');
    }
});
