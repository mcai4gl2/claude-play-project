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
@click.pass_context
def query(ctx, query, top_k, template, no_sources, threshold, copy_ready):
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
        # Perform search
        results = query_engine.search_notes(
            query, 
            top_k=top_k, 
            similarity_threshold=threshold
        )
        
        if not results:
            click.echo("❌ No relevant notes found.")
            if threshold:
                click.echo(f"Try lowering the similarity threshold (current: {threshold})")
            return
        
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