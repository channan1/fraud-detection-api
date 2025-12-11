async function modelPerf(model){
    const dataCells = document.querySelectorAll(`.${model}`);
    const proba = document.getElementById('threshold').value / 100
    dataCells.forEach(cell => {
        cell.innerHTML = '';
        const spinner = document.createElement("div")
        spinner.id = 'loading-spinner'
        cell.appendChild(spinner)
    });

    try {
        start = performance.now()
        response = await fetch(`http://127.0.0.1:8000/metrics/`,{
            method: 'POST',
            headers: {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({model: `${model}`, proba:`${proba}`})
        })
        data = await response.json()
        end = performance.now()
        document.getElementById(`${model}-sc`).textContent = response.status
        document.getElementById(`${model}-acc`).textContent = data.accuracy
        document.getElementById(`${model}-prec`).textContent = data.precision
        document.getElementById(`${model}-rec`).textContent = data.recall
        document.getElementById(`${model}-f1`).textContent = data.f1
        document.getElementById(`${model}-roc_auc`).textContent = data.roc_auc
        document.getElementById(`${model}-tp`).textContent = data.conf_mat[3]
        document.getElementById(`${model}-fp`).textContent = data.conf_mat[1]
        document.getElementById(`${model}-tn`).textContent = data.conf_mat[0]
        document.getElementById(`${model}-fn`).textContent = data.conf_mat[2]
        document.getElementById(`${model}-rt`).textContent = parseFloat(end-start).toFixed(2)
        return data
    } catch (error) {
        console.error("Error fetching data", error)
    }
}

async function modelPerfMult(){
    const elements = document.getElementsByClassName('modelName')
    for (let i = 0; i <elements.length; i++){
        modelPerf(elements[i].textContent)
    }
}

async function modelQuery(model){
        try {
        await fetch(`http://127.0.0.1:8000/metrics/`,{
            method: 'POST',
            headers: {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"model":`${model}`})
        })
        .then(response => response.json())
        .then(data => {return data})
    } catch (error) {
        console.error("Error fetching data", error)
    }
}

function updateSlider() {
    const slider = document.getElementById('threshold');
    const display = document.getElementById('displayValue');
    let formattedNum = slider.value / 100
    display.textContent = formattedNum.toFixed(2);
}

function randVal() {
    value = Math.floor(Math.random() * 49999)
    return value
}

async function singleQuery(){

    const model = document.getElementById('models').value
    const proba = document.getElementById('threshold').value / 100
    const transaction = document.getElementById('predictors').value

    const startTime = performance.now()
    try {
        const response = await fetch(`http://127.0.0.1:8000/obs_predict/`, {
            method: 'POST',
            headers: {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({model:`${model}`, index:`${transaction}`, proba: `${proba}`})
        })
        const data = await response.json();

        const endTime = performance.now()

        const duration = endTime - startTime
        const duration_formatted = duration.toFixed(2)
        if (!response.ok){
            displayApiStatus(response.status, data.detail)
            return {"status":response.status, "data":data.detail, "duration":duration_formatted}
            throw new Error('Network response was not ok.')
        }
        displayApiStatus(response.status)
        displayApiResults(data)
        return {"status":response.status, "data":data, "duration":duration_formatted}
        
    } catch (error) {
        console.error('Network Error:', error)
    }
}

function displayApiResults(data) {
    const resultsContainer = document.getElementById('api-results');
    resultsContainer.innerHTML = '';
    let itemDiv = document.createElement('div');
    const tableContainer = document.getElementById('table')
    tableContainer.innerHTML = '';
    if (typeof(data)=='string'){
        collection = `<h3 font-family:'Courier New', Courier, monospace;>Error: Index Out of Range (0 <= index < 5000) </h3>`
        itemDiv.innerHTML = collection
        resultsContainer.appendChild(itemDiv)
    } else {
        if(data.prediction == 0){
            pred_verbose = "Not Fraud"
            pred_class = "not-fraud"
        }else{
            pred_verbose = "Fraud"
            pred_class = "fraud"
        }
        if(data.actual == 0) {
            act_verbose = "Not Fraud"
            act_class = "not-fraud"
        }else{
            act_verbose = "Fraud"
            act_class = "fraud"
        }
        collection = `
            <h3 font-family:'Courier New', Courier, monospace;>Prediction: <div class=${pred_class}>${pred_verbose}</div> (${data.prediction_proba} Risk)</h3>
            
            <h3 font-family:'Courier New', Courier, monospace;>Actual: <div class=${act_class}>${act_verbose}</div></h3>
        `
        itemDiv.innerHTML = collection
        resultsContainer.appendChild(itemDiv)
        let tableDiv = document.createElement('div')
        collection = '';
        collection = collection + '<table> <thead> <tr>'
        const arr_to_loop = data.headers
        collection = collection + `<th>Feature</th><th>Value</tr></thead><tbody>`
        c = 0
        for (const item of arr_to_loop) {
            collection = collection + `<tr><td>${item}</td><td>${data.observation[c]}</td></tr>`
            c = c + 1
        }
        collection = collection + "</tbody></table>"
        collection = collection + "<footer>Fraud_Label: **1 = Fraud, 0 = No Fraud</footer>"
        tableDiv.innerHTML = collection
        tableContainer.appendChild(tableDiv)
    }
}

async function singleQueryTimed() {
    response = await singleQuery();
    console.log(response.status)
    timeToRun = `Ran in ${response.duration} ms`
    const resultContainer = document.getElementById('timer')
    resultContainer.innerHTML = '';
    let itemDiv = document.createElement('div');
    itemDiv.innerHTML = timeToRun
    resultContainer.appendChild(itemDiv)
    const divElement = document.getElementById('results-box')
    if (divElement){
        if (response.status == 200){
            divElement.className = "float-right-visible"
        }else{
            divElement.className= 'float-right'
        }
    }
}

function displayApiStatus(status, detail) {
    const parentContainer = document.getElementById('status')
    removeChildNodes(parentContainer)
    const newDiv = document.createElement('div');
    if (status==200){
        newDiv.innerHTML = `Response Code: <p id="status-p">${status}</p>`
    }else{
        newDiv.innerHTML = `Response Code: <p id="status-p">${status}: ${detail}</p>`
    }
    parentContainer.appendChild(newDiv)
    const statusContainer = document.getElementById('status-p')

    if (status >= 200 && status < 300){
        statusContainer.classList.add('status-success')
    } else if (status >= 400 && status < 500){
        statusContainer.classList.add('status-client-error')
        const resultsContainer = document.getElementById('api-results');
        resultsContainer.innerHTML = '';
        let itemDiv = document.createElement('div');
        const tableContainer = document.getElementById('table')
        tableContainer.innerHTML = '';
    } else if (status >= 500 && status < 600) {
        statusContainer.classList.add('status-server-error')
    } else {
        statusContainer.classList.add('status-server-default')
    }

}

function descriptorFilter(value) {
    const selection = document.querySelectorAll('.descClass')
    selection.forEach(element=> {
        if (value == "All Descriptive Analytics"){
            element.style.display = 'block';
        }else{
            element.style.display = 'none';
        }
    })
    if (value != "All Descriptive Analytics") {
        obj = {
            "Correlation Matrices":"corr",
            "Scales": "scales",
            "Fraud by Key Attributes": "fbka",
            "Time Series Analysis": "tsa",
            "Principal Component Analysis": "pca"
        }
        const selectionChange = document.getElementById(obj[value])
        selectionChange.style.display = 'block'
    }
}

function removeChildNodes(parent) {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
}

function resetSlider(){
    const slider = document.getElementById('threshold')
    slider.value = 50
    updateSlider()
}

window.onload = function(){
    if (window.location.pathname==='/descriptors') {
        descriptorFilter("Correlation Matrices")
    }else if (window.location.pathname=='/query' || window.location.pathname == '/performance'){
        updateSlider()
    }
}
