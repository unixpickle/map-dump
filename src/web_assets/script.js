const MAX_SUGGESTIONS = 50;
const DOWN_KEY = 40;
const UP_KEY = 38;
const ENTER_KEY = 13;

// State persisted across reloads of the search results.
let LATEST_EMBEDDING = '0';
let MAP_VIEWER_OPEN = true;

class App {
    constructor() {
        this.searchBox = new SearchBox();
        this.historyManager = new HistoryManager();
        this.results = document.getElementById('results');
        this.resultsError = document.getElementById('results-error');
        this.resultsData = document.getElementById('results-data');

        this.searchBox.onSearch = (query) => {
            this.searchQuery(query);
            this.historyManager.update(query);
        };
        this.historyManager.onChange = (query) => this.showNewQuery(query);
        if (this.historyManager.query) {
            this.showNewQuery(this.historyManager.query);
        } else {
            this.searchBox.focus();
        }
    }

    async searchQuery(query) {
        this.results.classList.remove('error');
        this.results.classList.add('loading');
        this.searchBox.setLoading(true);

        const cleanup = () => {
            this.searchBox.setLoading(false);
            this.results.classList.remove('loading');
        };

        let table;
        try {
            table = await createNeighborTable(query, (q) => {
                this.showNewQuery(q);
                this.historyManager.update(q);
            });
            if (query !== this.searchBox.getQuery()) {
                // A new query could have been run due to a popstate.
                return;
            }
        } catch (e) {
            this.results.classList.add('error');
            this.resultsError.textContent = e + '';
            cleanup();
            return;
        }
        cleanup();

        this.resultsData.textContent = '';
        this.resultsData.appendChild(table);
    };

    showNewQuery(query) {
        if (!query) {
            this.clearQuery();
        } else {
            this.searchBox.setQuery(query);
            return this.searchQuery(query);
        }
    }

    clearQuery() {
        this.results.classList.remove('error', 'loading');
        this.resultsData.textContent = '';
        this.searchBox.setQuery('');
        this.searchBox.setLoading(false);
        this.searchBox.focus();
    }
}

class SearchBox {
    constructor() {
        this.element = document.getElementById('search-container');
        this.input = document.getElementById('search-box');
        this.clearButton = document.getElementById('search-clear-button');
        this.suggestionContainer = document.getElementById('suggestions');
        this.suggestionElements = [];
        this.curSuggestion = 0;
        this.onSearch = (_) => null;

        this.input.addEventListener('keydown', (e) => this._keyDown(e));
        this.input.addEventListener('input', (e) => setTimeout(() => this._querySuggestions(), 10));
        this.input.addEventListener('focus', () => this._updateFocus());
        this.input.addEventListener('blur', () => {
            // Delay allows clicking on a suggestion.
            setTimeout(() => this._updateFocus(), 500);
        });

        this.clearButton.addEventListener('click', () => {
            this.input.value = '';
            this._querySuggestions();

            // Clicking the button could have made focus leave the input,
            // which would hide the suggestion box (and keyboard on mobile).
            this.input.focus();
        });

        this.suggestor = new LocationSuggestor();
        this.suggestor.onSuggestions = (q, results) => this._populateSuggestions(q, results);

        this._querySuggestions();
    }

    focus() {
        this.input.focus();
        this._updateFocus();
    }

    blur() {
        this.input.blur();
        this._updateFocus();
    }

    setLoading(f) {
        // TODO: disable search while results are loading.
    }

    getQuery() {
        return this.input.value;
    }

    setQuery(q) {
        this.input.value = q;
        this._querySuggestions();
    }

    _updateFocus() {
        if (document.activeElement === this.input) {
            this.element.classList.add('active');
        } else {
            this.element.classList.remove('active');
        }
    }

    _keyDown(e) {
        if (e.which == DOWN_KEY || e.which == UP_KEY) {
            const direction = (e.which == DOWN_KEY ? 1 : -1);
            const newCur = this.curSuggestion + direction;
            this._selectSuggestion(newCur, true);
            e.preventDefault();
            e.stopPropagation();
            return false;
        } else if (e.which == ENTER_KEY) {
            this._searchCurrent();
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        return true;
    }

    _selectSuggestion(idx, shouldScroll) {
        if (idx != this.curSuggestion && idx < this.suggestionElements.length && idx >= 0) {
            this.suggestionElements[this.curSuggestion].classList.remove('suggestion-cur');
            this.curSuggestion = idx;
            const curEl = this.suggestionElements[this.curSuggestion];
            curEl.classList.add('suggestion-cur');
            if (shouldScroll) {
                this._scrollToSuggestion(curEl);
            }
        }
    }

    _scrollToSuggestion(sugg) {
        const offsetY = sugg.getBoundingClientRect().top - this.suggestionContainer.getBoundingClientRect().top;
        const maxY = this.suggestionContainer.offsetHeight - sugg.offsetHeight;
        if (offsetY < 0) {
            this.suggestionContainer.scrollTop += offsetY;
        } else if (offsetY > maxY) {
            this.suggestionContainer.scrollTop += (offsetY - maxY);
        }
    }

    _searchCurrent() {
        if (this.suggestionElements.length) {
            this.input.value = this.suggestionElements[this.curSuggestion].textContent;
            this._querySuggestions();
        }

        // Hide suggestions faster than `blur` allows.
        this.input.blur();
        this._updateFocus();

        this.onSearch(this.input.value);
    }

    _querySuggestions() {
        this.suggestor.lookup(this.input.value);
    }

    _populateSuggestions(query, results) {
        if (this.input.value !== query) {
            return;
        }

        this.suggestionElements = [];
        this.suggestionContainer.textContent = '';

        if (!results.length) {
            this.suggestionContainer.classList.add('empty');
        } else {
            this.suggestionContainer.classList.remove('empty');
        }

        results.slice(0, MAX_SUGGESTIONS).forEach((x, i) => {
            const el = document.createElement('div');
            el.textContent = x;
            el.className = 'suggestion' + (i == 0 ? ' suggestion-cur' : '');
            this.suggestionContainer.appendChild(el);
            this.suggestionElements.push(el);

            el.addEventListener('mouseover', () => {
                this._selectSuggestion(i, false);
            });
            el.addEventListener('click', () => {
                this._selectSuggestion(i, true);
                this._searchCurrent();
            });
        });
        this.curSuggestion = 0;
    }
}

class LocationSuggestor {
    constructor() {
        this._names = null;
        this._counts = null;

        this.onSuggestions = (_q, _r) => null;
        this._queryID = 0;
    }

    async lookup(query) {
        const queryID = ++this._queryID;

        // Prioritize results.
        const hasPrefix = [];
        const contains = [];

        const lowerQuery = query.toLowerCase();
        (await this._getNamesAndCounts()).forEach((nameCount) => {
            const [name, count] = nameCount;
            const lower = name.toLowerCase();
            if (lower.startsWith(lowerQuery)) {
                hasPrefix.push([name, count]);
            } else if (lower.includes(lowerQuery)) {
                contains.push([name, count]);
            }
        });
        hasPrefix.sort((x, y) => y[1] - x[1]);
        contains.sort((x, y) => y[1] - x[1]);
        const results = hasPrefix.concat(contains).map((pair) => pair[0]);
        if (queryID == this._queryID) {
            this.onSuggestions(query, results);
        }
    }

    async _getNamesAndCounts() {
        if (this._names === null) {
            const results = await (await fetch('/api?f=stores')).json();
            this._names = results.names;
            this._counts = results.counts;
        }
        return this._names.map((x, i) => [x, this._counts[i]]);
    }
}

class HistoryManager {
    constructor() {
        this.query = this._parseLocation();
        this.onChange = (_) => null;

        window.addEventListener('popstate', () => {
            const newLocation = this._parseLocation();
            if (newLocation !== this.query) {
                this.query = newLocation;
                this.onChange(newLocation);
            }
        });
    }

    update(query) {
        if (query === this.query) {
            return;
        }
        this.query = query;
        if (!this.query) {
            history.pushState({}, '', '');
        } else {
            history.pushState({}, '', '#' + encodeURIComponent(this.query));
        }
    }

    _parseLocation() {
        if (location.hash) {
            return decodeURIComponent(location.hash.slice(1)) || null;
        } else {
            return null;
        }
    }
}

async function createNeighborTable(name, onQuery) {
    const numResults = 20;
    const url = '/api?f=knn&count=' + numResults + '&q=' + encodeURIComponent(name);
    const results = await (await fetch(url)).json();
    if (results['error']) {
        throw results['error'];
    } else {
        const embNames = Object.keys(results.results);
        embNames.sort();
        const tables = embNames.map((name) => {
            const table = document.createElement('table');
            const header = document.createElement('thead');
            const headerRow = document.createElement('tr');
            ['Name', 'Dot'].forEach((x) => {
                const th = document.createElement('th');
                th.textContent = x;
                headerRow.appendChild(th);
            });
            header.appendChild(headerRow);
            table.appendChild(header);

            const body = document.createElement('tbody');
            results.results[name].forEach((store, i) => {
                const dot = results.dots[name][i].toFixed(5);

                const tr = document.createElement('tr');
                [store, dot].forEach((x, i) => {
                    const td = document.createElement('td');
                    if (i == 0) {
                        // This is a location name, so it should hyperlink
                        // to results for the name.
                        const link = document.createElement('a');
                        link.addEventListener("click", () => onQuery(x));
                        link.textContent = x;
                        td.appendChild(link);
                    } else {
                        td.textContent = x;
                    }
                    tr.appendChild(td);
                });
                body.appendChild(tr);
            });
            table.appendChild(body);
            return table;
        });
        const switcher = tableSwitcher(embNames, tables);
        const mapViewer = createMapViewer(name, results);
        const container = document.createElement('div');
        container.className = 'results-container';
        container.appendChild(mapViewer);
        container.appendChild(switcher);
        return container;
    }
}

function tableSwitcher(names, tables) {
    const element = document.createElement('div');
    element.className = 'table-switcher';
    const switchControls = document.createElement('div');
    switchControls.className = 'table-switcher-controls';
    const label = document.createElement('label');
    label.textContent = 'Embeddings:';
    switchControls.appendChild(label);
    const select = document.createElement('select');
    names.forEach((x, i) => {
        const option = document.createElement('option');
        option.value = i + '';
        option.textContent = x;
        select.appendChild(option);
    });
    select.value = LATEST_EMBEDDING;
    select.addEventListener('input', () => LATEST_EMBEDDING = select.value);
    switchControls.appendChild(select);
    const downloadLink = document.createElement('a');
    downloadLink.href = '/vecs.json';
    downloadLink.className = 'download-link';
    downloadLink.textContent = '(Download All)';
    switchControls.appendChild(downloadLink);
    element.appendChild(switchControls);

    let curTable = null;
    const showSelected = () => {
        if (curTable) {
            element.removeChild(curTable);
        }
        curTable = tables[parseInt(select.value)];
        element.appendChild(curTable);
    };

    select.addEventListener('change', showSelected);
    showSelected();

    return element;
}

function createMapViewer(name, results) {
    const mapViewer = document.createElement('details');
    mapViewer.className = 'map-viewer';

    const countLabel = document.createElement('summary');
    countLabel.className = 'count-label';
    countLabel.textContent = 'Found ' + results['store_count'] + ' locations named "' + results['query'] + '".'
    mapViewer.appendChild(countLabel);

    const mapImage = document.createElement('img');
    mapImage.className = 'map-image';
    mapImage.src = '/map?q=' + encodeURIComponent(name);
    mapViewer.appendChild(mapImage);

    mapViewer.open = MAP_VIEWER_OPEN;
    mapViewer.addEventListener('toggle', () => MAP_VIEWER_OPEN = mapViewer.open);

    return mapViewer;
}

window.addEventListener('load', () => window.app = new App());
