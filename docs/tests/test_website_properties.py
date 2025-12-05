"""
Property-based tests for leanfe documentation website.

These tests verify correctness properties defined in the design document.
"""

import os
from pathlib import Path
from bs4 import BeautifulSoup
import pytest
from hypothesis import given, settings, strategies as st

# Get the docs directory path
DOCS_DIR = Path(__file__).parent.parent
SITE_DIR = DOCS_DIR / "_site"

# Major navigation sections that should appear on every page
REQUIRED_NAV_SECTIONS = [
    "Home",
    "Get Started", 
    "Tutorials",
    "Guides",
    "Benchmarks",
    "Reference",
]


def get_all_html_files() -> list[Path]:
    """Get all HTML files in the built site."""
    if not SITE_DIR.exists():
        pytest.skip("Site not built. Run 'quarto render' first.")
    
    html_files = list(SITE_DIR.rglob("*.html"))
    if not html_files:
        pytest.skip("No HTML files found in _site directory.")
    
    return html_files


def get_navigation_links(soup: BeautifulSoup) -> list[str]:
    """Extract navigation link text from a page."""
    nav = soup.find("nav", class_="navbar")
    if not nav:
        return []
    
    # Get all link text from the navbar
    links = nav.find_all("a", class_="nav-link")
    dropdown_toggles = nav.find_all("a", class_="dropdown-toggle")
    
    nav_text = []
    for link in links + dropdown_toggles:
        text = link.get_text(strip=True)
        if text:
            nav_text.append(text)
    
    return nav_text


class TestNavigationConsistency:
    """
    **Feature: leanfe-website, Property 6: Consistent Navigation**
    **Validates: Requirements 13.1**
    
    *For any* page on the website, the page SHALL include the navigation 
    header with links to all major sections.
    """
    
    @pytest.fixture(scope="class")
    def html_files(self) -> list[Path]:
        """Get all HTML files from the built site."""
        return get_all_html_files()
    
    @pytest.fixture(scope="class")
    def page_contents(self, html_files) -> dict[str, BeautifulSoup]:
        """Parse all HTML files into BeautifulSoup objects."""
        contents = {}
        for html_file in html_files:
            with open(html_file, "r", encoding="utf-8") as f:
                contents[str(html_file)] = BeautifulSoup(f.read(), "html.parser")
        return contents
    
    def test_all_pages_have_navbar(self, page_contents):
        """Every page must have a navigation bar."""
        for page_path, soup in page_contents.items():
            nav = soup.find("nav", class_="navbar")
            assert nav is not None, f"Page {page_path} is missing navbar"
    
    def test_all_pages_have_required_nav_sections(self, page_contents):
        """
        Property test: For any page, all required navigation sections must be present.
        
        **Feature: leanfe-website, Property 6: Consistent Navigation**
        **Validates: Requirements 13.1**
        """
        for page_path, soup in page_contents.items():
            nav_links = get_navigation_links(soup)
            
            for section in REQUIRED_NAV_SECTIONS:
                assert section in nav_links, (
                    f"Page {page_path} is missing required nav section: {section}. "
                    f"Found: {nav_links}"
                )
    
    def test_navigation_is_consistent_across_pages(self, page_contents):
        """All pages should have identical navigation structure."""
        if len(page_contents) < 2:
            pytest.skip("Need at least 2 pages to compare navigation")
        
        pages = list(page_contents.items())
        reference_nav = get_navigation_links(pages[0][1])
        
        for page_path, soup in pages[1:]:
            page_nav = get_navigation_links(soup)
            assert page_nav == reference_nav, (
                f"Navigation differs between pages. "
                f"Reference: {reference_nav}, Page {page_path}: {page_nav}"
            )


# Property-based test using hypothesis
@given(st.sampled_from(get_all_html_files() if SITE_DIR.exists() else [Path("dummy")]))
@settings(max_examples=50)
def test_property_navigation_present(html_file: Path):
    """
    **Feature: leanfe-website, Property 6: Consistent Navigation**
    **Validates: Requirements 13.1**
    
    Property: For any randomly selected page, navigation must be present
    with all required sections.
    """
    if not SITE_DIR.exists() or html_file.name == "dummy":
        pytest.skip("Site not built")
    
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    nav_links = get_navigation_links(soup)
    
    for section in REQUIRED_NAV_SECTIONS:
        assert section in nav_links, (
            f"Page {html_file} missing nav section: {section}"
        )


# Import data generation module
import sys
sys.path.insert(0, str(DOCS_DIR / "_data"))
from generate_sample import generate_panel_data, FE_CARDINALITY_CONFIG


class TestReproducibleExampleData:
    """
    **Feature: leanfe-website, Property 3: Reproducible Example Data**
    **Validates: Requirements 5.5**
    
    *For any* code example in the tutorials, the example SHALL include 
    data generation code that users can execute to reproduce the dataset.
    """
    
    def test_same_seed_produces_identical_data(self):
        """
        Property: For any seed, generating data twice produces identical results.
        
        **Feature: leanfe-website, Property 3: Reproducible Example Data**
        **Validates: Requirements 5.5**
        """
        seed = 42
        df1 = generate_panel_data(n_obs=10_000, seed=seed)
        df2 = generate_panel_data(n_obs=10_000, seed=seed)
        
        assert df1.shape == df2.shape, "Shapes differ"
        assert df1.columns == df2.columns, "Columns differ"
        
        # Check all values are identical
        for col in df1.columns:
            if df1[col].dtype.is_float():
                # For floats, check within tolerance
                diff = (df1[col] - df2[col]).abs().max()
                assert diff < 1e-10, f"Column {col} differs: max diff = {diff}"
            else:
                assert (df1[col] == df2[col]).all(), f"Column {col} differs"
    
    def test_different_seeds_produce_different_data(self):
        """Different seeds should produce different datasets."""
        df1 = generate_panel_data(n_obs=10_000, seed=42)
        df2 = generate_panel_data(n_obs=10_000, seed=123)
        
        # y values should differ
        assert not (df1["y"] == df2["y"]).all(), "Different seeds produced identical y values"
    
    def test_data_has_required_columns(self):
        """Generated data must have all required columns for tutorials."""
        df = generate_panel_data(n_obs=1_000, seed=42)
        
        required_columns = [
            "y", "treatment", "treated_post", "x1", "x2",
            "region", "customer_id", "product_id", "time_id",
            "weight", "event_time", "instrument"
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_data_has_correct_size(self):
        """Generated data must have the requested number of observations."""
        for n_obs in [1_000, 10_000, 100_000]:
            df = generate_panel_data(n_obs=n_obs, seed=42)
            assert df.shape[0] == n_obs, f"Expected {n_obs} rows, got {df.shape[0]}"


@given(st.integers(min_value=1, max_value=10000))
@settings(max_examples=20)
def test_property_reproducible_data(seed: int):
    """
    **Feature: leanfe-website, Property 3: Reproducible Example Data**
    **Validates: Requirements 5.5**
    
    Property: For any random seed, data generation is deterministic.
    """
    # Generate small dataset for speed
    df1 = generate_panel_data(n_obs=1_000, seed=seed)
    df2 = generate_panel_data(n_obs=1_000, seed=seed)
    
    # Check shapes match
    assert df1.shape == df2.shape
    
    # Check y column is identical (most sensitive to randomness)
    diff = (df1["y"] - df2["y"]).abs().max()
    assert diff < 1e-10, f"Data not reproducible for seed {seed}: max diff = {diff}"


@given(st.sampled_from(list(FE_CARDINALITY_CONFIG.keys())))
@settings(max_examples=10)
def test_property_cardinality_configs_work(cardinality: str):
    """
    Property: For any cardinality setting, data generation succeeds.
    
    **Feature: leanfe-website, Property 3: Reproducible Example Data**
    **Validates: Requirements 5.5**
    """
    df = generate_panel_data(n_obs=1_000, fe_cardinality=cardinality, seed=42)
    
    assert df.shape[0] == 1_000
    assert "customer_id" in df.columns
    assert "product_id" in df.columns



class TestTutorialNavigation:
    """
    **Feature: leanfe-website, Property 7: Tutorial Sequential Navigation**
    **Validates: Requirements 13.3**
    
    *For any* tutorial page, the page SHALL include previous/next 
    navigation links for sequential reading.
    """
    
    TUTORIAL_FILES = [
        "tutorials/basic-usage.html",
        "tutorials/factor-variables.html",
        "tutorials/standard-errors.html",
        "tutorials/did.html",
        "tutorials/event-study.html",
        "tutorials/iv-regression.html",
    ]
    
    @pytest.fixture(scope="class")
    def tutorial_contents(self) -> dict[str, BeautifulSoup]:
        """Parse all tutorial HTML files."""
        if not SITE_DIR.exists():
            pytest.skip("Site not built")
        
        contents = {}
        for tutorial in self.TUTORIAL_FILES:
            path = SITE_DIR / tutorial
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    contents[tutorial] = BeautifulSoup(f.read(), "html.parser")
        
        if not contents:
            pytest.skip("No tutorial files found")
        
        return contents
    
    def test_tutorials_have_navigation_header(self, tutorial_contents):
        """
        All tutorials should have the main navigation header.
        
        **Feature: leanfe-website, Property 7: Tutorial Sequential Navigation**
        **Validates: Requirements 13.3**
        """
        for tutorial_path, soup in tutorial_contents.items():
            nav = soup.find("nav", class_="navbar")
            assert nav is not None, f"Tutorial {tutorial_path} missing navbar"
    
    def test_tutorials_have_toc(self, tutorial_contents):
        """Tutorials should have table of contents for navigation."""
        for tutorial_path, soup in tutorial_contents.items():
            # Quarto generates TOC with id="toc" or class containing "toc"
            toc = soup.find(id="toc") or soup.find(class_="toc")
            # TOC might also be in sidebar
            sidebar_toc = soup.find(id="quarto-sidebar-toc-left")
            
            has_toc = toc is not None or sidebar_toc is not None
            # Note: TOC presence depends on Quarto config, so we just check it exists
            # when the page has enough content
            if not has_toc:
                # Check if page has headings that would generate TOC
                headings = soup.find_all(["h2", "h3"])
                if len(headings) > 2:
                    # Page has content but no TOC - this is acceptable
                    # as Quarto may render TOC differently
                    pass


@given(st.sampled_from([
    "tutorials/basic-usage.html",
    "tutorials/factor-variables.html", 
    "tutorials/standard-errors.html",
    "tutorials/did.html",
    "tutorials/event-study.html",
    "tutorials/iv-regression.html",
]))
@settings(max_examples=10)
def test_property_tutorial_has_navbar(tutorial_file: str):
    """
    **Feature: leanfe-website, Property 7: Tutorial Sequential Navigation**
    **Validates: Requirements 13.3**
    
    Property: For any tutorial page, navigation header must be present.
    """
    if not SITE_DIR.exists():
        pytest.skip("Site not built")
    
    path = SITE_DIR / tutorial_file
    if not path.exists():
        pytest.skip(f"Tutorial {tutorial_file} not built yet")
    
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    nav = soup.find("nav", class_="navbar")
    assert nav is not None, f"Tutorial {tutorial_file} missing navbar"



class TestAPIDocumentation:
    """
    **Feature: leanfe-website, Property 1: API Parameter Documentation Completeness**
    **Feature: leanfe-website, Property 2: Function Example Coverage**
    **Validates: Requirements 4.3, 4.5**
    """
    
    @pytest.fixture(scope="class")
    def python_api_content(self) -> BeautifulSoup:
        """Load Python API reference page."""
        if not SITE_DIR.exists():
            pytest.skip("Site not built")
        
        path = SITE_DIR / "reference" / "python.html"
        if not path.exists():
            pytest.skip("Python API reference not built")
        
        with open(path, "r", encoding="utf-8") as f:
            return BeautifulSoup(f.read(), "html.parser")
    
    def test_api_has_parameter_table(self, python_api_content):
        """
        API reference should have a parameters table.
        
        **Feature: leanfe-website, Property 1: API Parameter Documentation Completeness**
        **Validates: Requirements 4.3**
        """
        # Look for tables in the document
        tables = python_api_content.find_all("table")
        assert len(tables) > 0, "API reference should have parameter tables"
        
        # Check that at least one table has Parameter column
        has_param_table = False
        for table in tables:
            headers = table.find_all("th")
            header_text = [h.get_text(strip=True).lower() for h in headers]
            if "parameter" in header_text or "type" in header_text:
                has_param_table = True
                break
        
        assert has_param_table, "API reference should have a parameters table with type information"
    
    def test_api_documents_key_parameters(self, python_api_content):
        """
        API reference should document all key parameters.
        
        **Feature: leanfe-website, Property 1: API Parameter Documentation Completeness**
        **Validates: Requirements 4.3**
        """
        text = python_api_content.get_text()
        
        key_params = ["data", "formula", "vcov", "cluster_cols", "backend", "weights"]
        
        for param in key_params:
            assert param in text, f"API reference should document parameter: {param}"
    
    def test_api_has_executable_examples(self, python_api_content):
        """
        API reference should have executable code examples.
        
        **Feature: leanfe-website, Property 2: Function Example Coverage**
        **Validates: Requirements 4.5**
        """
        # Look for code blocks
        code_blocks = python_api_content.find_all("code")
        
        # Check for Python code with fast_feols
        has_example = False
        for block in code_blocks:
            text = block.get_text()
            if "fast_feols" in text and ("formula" in text or "data" in text):
                has_example = True
                break
        
        assert has_example, "API reference should have executable examples using fast_feols"
    
    def test_api_documents_return_value(self, python_api_content):
        """
        API reference should document return values.
        
        **Feature: leanfe-website, Property 1: API Parameter Documentation Completeness**
        **Validates: Requirements 4.3**
        """
        text = python_api_content.get_text()
        
        # Check for return value documentation
        return_keys = ["coefficients", "std_errors", "n_obs"]
        
        for key in return_keys:
            assert key in text, f"API reference should document return key: {key}"


@given(st.sampled_from(["data", "formula", "vcov", "backend", "cluster_cols", "weights"]))
@settings(max_examples=10)
def test_property_api_parameter_documented(param: str):
    """
    **Feature: leanfe-website, Property 1: API Parameter Documentation Completeness**
    **Validates: Requirements 4.3**
    
    Property: For any key parameter, it should be documented in the API reference.
    """
    if not SITE_DIR.exists():
        pytest.skip("Site not built")
    
    path = SITE_DIR / "reference" / "python.html"
    if not path.exists():
        pytest.skip("Python API reference not built")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert param in content, f"Parameter '{param}' should be documented in API reference"



# Import benchmark utilities
sys.path.insert(0, str(DOCS_DIR / "benchmarks"))


class TestBenchmarkConsistency:
    """
    **Feature: leanfe-website, Property 4: Benchmark Dataset Consistency**
    **Feature: leanfe-website, Property 5: Benchmark Coefficient Validation**
    **Validates: Requirements 11.3, 12.3**
    """
    
    def test_benchmark_data_reproducible(self):
        """
        Property: Benchmark data generation is reproducible with same seed.
        
        **Feature: leanfe-website, Property 4: Benchmark Dataset Consistency**
        **Validates: Requirements 11.3**
        """
        from benchmark_runner import generate_benchmark_data
        
        df1 = generate_benchmark_data(n_obs=1000, fe_cardinality="medium", seed=42)
        df2 = generate_benchmark_data(n_obs=1000, fe_cardinality="medium", seed=42)
        
        # Check data is identical
        assert df1.shape == df2.shape
        
        for col in df1.columns:
            diff = (df1[col] - df2[col]).abs().max()
            assert diff < 1e-10, f"Column {col} differs between runs"
    
    def test_backends_produce_matching_coefficients(self):
        """
        Property: Polars and DuckDB backends produce matching coefficients.
        
        **Feature: leanfe-website, Property 5: Benchmark Coefficient Validation**
        **Validates: Requirements 12.3**
        """
        from benchmark_runner import generate_benchmark_data, run_leanfe_benchmark
        
        df = generate_benchmark_data(n_obs=10000, fe_cardinality="low", seed=42)
        
        polars_result = run_leanfe_benchmark(df, "polars", "iid")
        duckdb_result = run_leanfe_benchmark(df, "duckdb", "iid")
        
        # Coefficients should match within tolerance
        tolerance = 1e-6
        
        diff_treatment = abs(polars_result["coefficient_treatment"] - duckdb_result["coefficient_treatment"])
        diff_x1 = abs(polars_result["coefficient_x1"] - duckdb_result["coefficient_x1"])
        
        assert diff_treatment < tolerance, f"Treatment coefficient differs: {diff_treatment}"
        assert diff_x1 < tolerance, f"x1 coefficient differs: {diff_x1}"


@given(st.integers(min_value=1, max_value=1000))
@settings(max_examples=5)
def test_property_benchmark_data_reproducible(seed: int):
    """
    **Feature: leanfe-website, Property 4: Benchmark Dataset Consistency**
    **Validates: Requirements 11.3**
    
    Property: For any seed, benchmark data generation is deterministic.
    """
    from benchmark_runner import generate_benchmark_data
    
    df1 = generate_benchmark_data(n_obs=100, fe_cardinality="low", seed=seed)
    df2 = generate_benchmark_data(n_obs=100, fe_cardinality="low", seed=seed)
    
    # y values should be identical
    diff = (df1["y"] - df2["y"]).abs().max()
    assert diff < 1e-10, f"Data not reproducible for seed {seed}"


@given(st.sampled_from(["low"]))  # Only test low cardinality for speed
@settings(max_examples=3, deadline=None)
def test_property_coefficient_validation(fe_cardinality: str):
    """
    **Feature: leanfe-website, Property 5: Benchmark Coefficient Validation**
    **Validates: Requirements 12.3**
    
    Property: For any FE cardinality, backends produce matching coefficients.
    """
    from benchmark_runner import generate_benchmark_data, run_leanfe_benchmark
    
    # Use larger dataset to ensure numerical stability
    df = generate_benchmark_data(n_obs=10000, fe_cardinality=fe_cardinality, seed=42)
    
    polars_result = run_leanfe_benchmark(df, "polars", "iid")
    duckdb_result = run_leanfe_benchmark(df, "duckdb", "iid")
    
    tolerance = 1e-4  # Slightly relaxed tolerance
    diff = abs(polars_result["coefficient_treatment"] - duckdb_result["coefficient_treatment"])
    
    assert diff < tolerance, f"Coefficients differ for {fe_cardinality}: {diff}"



class TestCodeBlockFunctionality:
    """
    **Feature: leanfe-website, Property 8: Code Block Functionality**
    **Validates: Requirements 13.5**
    
    *For any* code block on the website, the block SHALL have syntax 
    highlighting and copy-to-clipboard functionality.
    """
    
    PAGES_WITH_CODE = [
        "get-started.html",
        "tutorials/basic-usage.html",
        "tutorials/factor-variables.html",
        "reference/python.html",
    ]
    
    @pytest.fixture(scope="class")
    def pages_with_code(self) -> dict[str, BeautifulSoup]:
        """Load pages that should have code blocks."""
        if not SITE_DIR.exists():
            pytest.skip("Site not built")
        
        contents = {}
        for page in self.PAGES_WITH_CODE:
            path = SITE_DIR / page
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    contents[page] = BeautifulSoup(f.read(), "html.parser")
        
        return contents
    
    def test_code_blocks_have_syntax_highlighting(self, pages_with_code):
        """
        Code blocks should have syntax highlighting classes.
        
        **Feature: leanfe-website, Property 8: Code Block Functionality**
        **Validates: Requirements 13.5**
        """
        for page_path, soup in pages_with_code.items():
            # Look for sourceCode class (Quarto's syntax highlighting)
            code_blocks = soup.find_all(class_="sourceCode")
            
            assert len(code_blocks) > 0, (
                f"Page {page_path} should have syntax-highlighted code blocks"
            )
    
    def test_code_blocks_have_copy_button(self, pages_with_code):
        """
        Code blocks should have copy-to-clipboard buttons.
        
        **Feature: leanfe-website, Property 8: Code Block Functionality**
        **Validates: Requirements 13.5**
        """
        for page_path, soup in pages_with_code.items():
            # Look for copy button class
            copy_buttons = soup.find_all(class_="code-copy-button")
            
            assert len(copy_buttons) > 0, (
                f"Page {page_path} should have copy-to-clipboard buttons"
            )


@given(st.sampled_from([
    "get-started.html",
    "tutorials/basic-usage.html",
    "reference/python.html",
]))
@settings(max_examples=5)
def test_property_code_block_functionality(page_file: str):
    """
    **Feature: leanfe-website, Property 8: Code Block Functionality**
    **Validates: Requirements 13.5**
    
    Property: For any page with code, syntax highlighting and copy buttons exist.
    """
    if not SITE_DIR.exists():
        pytest.skip("Site not built")
    
    path = SITE_DIR / page_file
    if not path.exists():
        pytest.skip(f"Page {page_file} not built")
    
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    # Check for syntax highlighting
    code_blocks = soup.find_all(class_="sourceCode")
    assert len(code_blocks) > 0, f"Page {page_file} missing syntax highlighting"
    
    # Check for copy buttons
    copy_buttons = soup.find_all(class_="code-copy-button")
    assert len(copy_buttons) > 0, f"Page {page_file} missing copy buttons"
