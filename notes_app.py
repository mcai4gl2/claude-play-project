#!/usr/bin/env python3

import click
import os
import sys
from pathlib import Path

from query_engine import QueryEngine
from vector_store import VectorStore
from context_formatter import ContextFormatter


@click.group()
@click.option('--notes-dir', '-d', default='./sample_notes', 
              help='Directory containing notes (default: ./sample_notes)')
@click.option('--db-dir', default='./vector_db', 
              help='Vector database directory (default: ./vector_db)')
@click.pass_context
def cli(ctx, notes_dir, db_dir):
    """Notes Vector Database Application
    
    Index your notes and query them using semantic search.
    """
    ctx.ensure_object(dict)
    ctx.obj['notes_dir'] = notes_dir
    ctx.obj['db_dir'] = db_dir
    
    # Initialize components
    ctx.obj['vector_store'] = VectorStore(persist_directory=db_dir)
    ctx.obj['query_engine'] = QueryEngine(ctx.obj['vector_store'])
    ctx.obj['formatter'] = ContextFormatter()


@cli.command()
@click.option('--force', '-f', is_flag=True, 
              help='Force reindex even if notes already exist')
@click.pass_context
def index(ctx, force):
    """Index notes from the specified directory."""
    notes_dir = ctx.obj['notes_dir']
    query_engine = ctx.obj['query_engine']
    
    if not os.path.exists(notes_dir):
        click.echo(f"❌ Notes directory '{notes_dir}' does not exist.", err=True)
        sys.exit(1)
    
    click.echo(f"📂 Indexing notes from: {notes_dir}")
    
    try:
        result = query_engine.index_notes(notes_dir, force_reindex=force)
        
        status = result['status']
        if status == 'success':
            click.echo(f"✅ Successfully indexed {result['indexed_count']} notes")
        elif status == 'incremental_update':
            click.echo(f"✅ Added {result['indexed_count']} new notes")
            click.echo(f"📊 Total notes in index: {result['total_count']}")
        elif status == 'up_to_date':
            click.echo("ℹ️  Index is already up to date")
            click.echo(f"📊 Total notes: {result['total_count']}")
        elif status == 'no_notes_found':
            click.echo(f"⚠️  No supported notes found in {notes_dir}")
            click.echo("Supported formats: .txt, .md, .rst")
        
    except Exception as e:
        click.echo(f"❌ Error during indexing: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--top-k', '-k', default=5, 
              help='Number of results to return (default: 5)')
@click.option('--template', '-t', default='standard',
              type=click.Choice(['standard', 'detailed', 'minimal']),
              help='Context template style (default: standard)')
@click.option('--no-sources', is_flag=True,
              help='Exclude source information from output')
@click.option('--threshold', type=float,
              help='Similarity threshold (0.0-1.0)')
@click.option('--copy-ready', is_flag=True,
              help='Format output for easy copying to LLM')
@click.option('--mode', '-m', default='auto',
              type=click.Choice(['auto', 'traditional', 'semantic', 'hybrid']),
              help='Search mode (default: auto)')
@click.option('--explain', is_flag=True,
              help='Show detailed explanation of search results')
@click.pass_context
def query(ctx, query, top_k, template, no_sources, threshold, copy_ready, mode, explain):
    """Search notes and generate context prompt for LLM."""
    query_engine = ctx.obj['query_engine']
    formatter = ctx.obj['formatter']
    
    # Check if index exists
    stats = query_engine.get_stats()
    if stats['total_notes'] == 0:
        click.echo("❌ No notes found in index. Run 'index' command first.", err=True)
        sys.exit(1)
    
    click.echo(f"🔍 Searching for: {query}")
    
    try:
        # Perform search with hybrid mode support
        results = query_engine.search_notes(
            query, 
            top_k=top_k, 
            similarity_threshold=threshold,
            search_mode=mode
        )
        
        if not results:
            click.echo("❌ No relevant notes found.")
            if threshold:
                click.echo(f"Try lowering the similarity threshold (current: {threshold})")
            return
        
        # Show search mode information
        search_mode_used = results[0].get('search_mode', mode)
        if search_mode_used:
            mode_icons = {
                'auto': '🎯', 'traditional': '📝', 'semantic': '🧠', 
                'hybrid': '⚡', 'semantic_fallback': '🔄'
            }
            icon = mode_icons.get(search_mode_used, '🔍')
            click.echo(f"{icon} Search mode: {search_mode_used}")
            
            # Show additional info for hybrid searches
            if search_mode_used == 'hybrid' and results:
                fusion_method = results[0].get('fusion_method', 'unknown')
                click.echo(f"🔀 Fusion method: {fusion_method}")
        
        # Show explanation if requested
        if explain:
            click.echo("\n" + "="*50)
            click.echo("🔍 SEARCH EXPLANATION")
            click.echo("="*50)
            
            explanation = query_engine.explain_search_results(query, results)
            
            # Query analysis
            qa = explanation.get('query_analysis', {})
            click.echo(f"Query type: {qa.get('type', 'unknown')} (confidence: {qa.get('confidence', 0):.2f})")
            if qa.get('phrases'):
                click.echo(f"Phrases: {', '.join(qa['phrases'])}")
            if qa.get('terms'):
                click.echo(f"Terms: {', '.join(qa['terms'])}")
            if qa.get('filters'):
                click.echo(f"Filters: {qa['filters']}")
            
            # Result explanations
            click.echo(f"\nTop {min(3, len(results))} result explanations:")
            for res_exp in explanation.get('result_explanations', [])[:3]:
                click.echo(f"\nRank {res_exp['rank']}: {res_exp['document_id']}")
                click.echo(f"  Score: {res_exp['final_score']:.4f}")
                click.echo(f"  Mode: {res_exp['search_mode']}")
                
                if 'source_ranks' in res_exp:
                    click.echo(f"  Source ranks: {res_exp['source_ranks']}")
                if 'source_scores' in res_exp:
                    click.echo(f"  Source scores: {res_exp['source_scores']}")
            
            click.echo("="*50 + "\n")
        
        # Show search summary
        click.echo(f"📋 Found {len(results)} relevant notes:")
        summary = formatter.format_search_results_summary(results)
        click.echo(summary)
        click.echo()
        
        # Generate context prompt
        context_prompt = formatter.format_context_prompt(
            query, 
            results, 
            include_sources=not no_sources,
            template_style=template
        )
        
        if copy_ready:
            context_prompt = formatter.create_copy_ready_prompt(context_prompt)
        
        click.echo("📝 Generated context prompt:")
        click.echo("─" * 60)
        click.echo(context_prompt)
        
    except Exception as e:
        click.echo(f"❌ Error during search: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show comprehensive statistics about the notes index and search usage."""
    query_engine = ctx.obj['query_engine']
    
    try:
        stats = query_engine.get_search_statistics()
        
        click.echo("📊 NOTES DATABASE STATISTICS")
        click.echo("=" * 40)
        
        # Basic stats
        click.echo(f"Total notes: {stats['total_notes']}")
        click.echo(f"Collection: {stats['collection_name']}")
        click.echo(f"Database directory: {stats['persist_directory']}")
        
        # Hybrid search stats
        if 'hybrid_search' in stats:
            hybrid = stats['hybrid_search']
            config = hybrid.get('configuration', {})
            
            click.echo(f"\n🔍 HYBRID SEARCH STATUS")
            click.echo("-" * 30)
            click.echo(f"BM25 indexed: {'✅' if config.get('bm25_indexed') else '❌'}")
            click.echo(f"Default fusion method: {config.get('default_fusion_method', 'unknown')}")
            
            # Search statistics
            search_stats = hybrid.get('search_statistics', {})
            if search_stats.get('total_searches', 0) > 0:
                click.echo(f"\nTotal searches performed: {search_stats['total_searches']}")
                
                mode_usage = search_stats.get('mode_usage', {})
                if mode_usage:
                    click.echo("Search mode usage:")
                    for mode, count in mode_usage.items():
                        percentage = (count / search_stats['total_searches']) * 100
                        click.echo(f"  {mode}: {count} ({percentage:.1f}%)")
                
                avg_results = search_stats.get('average_results', {})
                if avg_results:
                    click.echo("Average results per search:")
                    for engine, avg in avg_results.items():
                        if avg > 0:
                            click.echo(f"  {engine}: {avg:.1f}")
            
            # Query processor stats
            qp_stats = hybrid.get('query_processor_stats', {})
            if qp_stats:
                click.echo(f"\n📝 QUERY PROCESSOR")
                click.echo("-" * 20)
                click.echo(f"Stop words: {qp_stats.get('stop_words_count', 0)}")
                click.echo(f"Technical indicators: {qp_stats.get('technical_indicators_count', 0)}")
                click.echo(f"Synonym entries: {qp_stats.get('synonym_entries', 0)}")
            
            # Fusion statistics
            fusion_stats = hybrid.get('fusion_statistics', {})
            if fusion_stats.get('total_fusions_performed', 0) > 0:
                click.echo(f"\n⚡ RESULT FUSION")
                click.echo("-" * 15)
                click.echo(f"Total fusions: {fusion_stats['total_fusions_performed']}")
                
                method_usage = fusion_stats.get('method_usage_counts', {})
                if method_usage:
                    click.echo("Fusion methods used:")
                    for method, count in method_usage.items():
                        click.echo(f"  {method}: {count}")
        
    except Exception as e:
        click.echo(f"❌ Error retrieving statistics: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show index statistics."""
    query_engine = ctx.obj['query_engine']
    
    try:
        stats = query_engine.get_stats()
        
        click.echo("📊 Index Statistics:")
        click.echo(f"   Total notes: {stats['total_notes']}")
        click.echo(f"   Collection: {stats['collection_name']}")
        click.echo(f"   Database: {stats['persist_directory']}")
        
    except Exception as e:
        click.echo(f"❌ Error getting stats: {e}", err=True)


@cli.command()
@click.pass_context
def clear(ctx):
    """Clear the entire index."""
    vector_store = ctx.obj['vector_store']
    
    if click.confirm('⚠️  This will delete all indexed notes. Continue?'):
        try:
            cleared_count = vector_store.clear_collection()
            click.echo(f"🗑️  Cleared {cleared_count} notes from index")
            
        except Exception as e:
            click.echo(f"❌ Error clearing index: {e}", err=True)
    else:
        click.echo("Operation cancelled.")


@cli.command()
@click.argument('title_query')
@click.option('--top-k', '-k', default=5,
              help='Number of results to return (default: 5)')
@click.pass_context
def search_title(ctx, title_query, top_k):
    """Search notes by title."""
    query_engine = ctx.obj['query_engine']
    formatter = ctx.obj['formatter']
    
    # Check if index exists
    stats = query_engine.get_stats()
    if stats['total_notes'] == 0:
        click.echo("❌ No notes found in index. Run 'index' command first.", err=True)
        sys.exit(1)
    
    click.echo(f"🔍 Searching titles for: {title_query}")
    
    try:
        results = query_engine.search_by_title(title_query, top_k)
        
        if not results:
            click.echo("❌ No matching titles found.")
            return
        
        click.echo(f"📋 Found {len(results)} matching notes:")
        
        for i, result in enumerate(results, 1):
            title = result['metadata']['title']
            filename = result['metadata']['filename']
            title_score = result.get('title_score', 0)
            relevance = result.get('relevance_score', 0)
            
            click.echo(f"{i}. **{title}** ({filename})")
            click.echo(f"   Title Score: {title_score:.2f}, Relevance: {relevance:.2f}")
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Error during title search: {e}", err=True)


if __name__ == '__main__':
    cli()