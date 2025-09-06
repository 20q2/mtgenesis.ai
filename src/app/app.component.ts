import { Component, ViewChild, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { catchError, finalize, Observable, of } from 'rxjs';
import { CardGenerationRequest } from './models/api.model';
import { Card, Rarity } from './models/card.model';
import { CardService } from './services/card.service';
import { HealthService, HealthStatus } from './services/health.service';
import { CardFormComponent } from './components/card-form/card-form.component';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  @ViewChild(CardFormComponent) cardFormComponent!: CardFormComponent;
  
  // Make Math available to template
  Math = Math;
  
  title = 'MTGenesis.AI';
  currentCard: Card = {
    name: 'New Card',
    manaCost: '{1}',
    type: 'Creature',
    colors: ['C'],
    cmc: 1,
    rarity: Rarity.COMMON,
    artPrompt: 'Fantasy art of new card',
    description: 'Your custom Magic: The Gathering card will appear here.'
  };
  
  // Loading states
  isGenerating = false;
  isGeneratingText = false;
  isGeneratingArt = false;
  generationProgress = 0;
  generationStartTime = 0;
  progressInterval: any = null;
  showExtendedMessage = false;
  
  // Message rotation properties
  currentMessage = { title: 'Generating Your Magic Card', subtitle: 'Creating artwork and card text with AI...' };
  generationMessages: any[] = [];
  messageInterval: any = null;
  messageVisible = true;
  
  // Error states
  error: string | null = null;
  textError: string | null = null;
  artError: string | null = null;
  
  // Success state
  successMessage: string | null = null;
  
  // Display toggle
  showCompleteCard: boolean = false;
  
  // Card transition state
  cardExiting: boolean = false;
  
  // Health check states
  isCheckingHealth = true;
  healthStatus: HealthStatus | null = null;
  healthError: string | null = null;
  modelsReady = false;
  showHealthBanner = false;
  
  constructor(
    private cardService: CardService, 
    private healthService: HealthService,
    private http: HttpClient
  ) {}

  ngOnInit(): void {
    // Add a delay before showing the health banner to prevent flash
    setTimeout(() => {
      this.showHealthBanner = true;
    }, 500);
    
    this.loadGenerationMessages();
    this.checkModelsHealth();
  }
  
  updateCard(card: Card): void {
    console.log('AppComponent: updateCard called with colors:', card.colors);
    
    // If we had a complete card and form changed, trigger exit animation
    if (this.currentCard.cardImageUrl && !card.cardImageUrl) {
      this.cardExiting = true;
      
      // After animation completes, update the card
      setTimeout(() => {
        this.currentCard = { ...card };
        this.cardExiting = false;
        console.log('AppComponent: currentCard updated to:', this.currentCard);
      }, 600); // Match animation duration
    } else {
      // No animation needed, update immediately
      this.currentCard = { ...card };
      console.log('AppComponent: currentCard updated to:', this.currentCard);
    }
    
    // Clear any previous errors or success messages when card is updated
    this.error = null;
    this.textError = null;
    this.artError = null;
    this.successMessage = null;
  }
  
  generateCard(card: Card): void {
    if (!this.modelsReady) {
      this.error = 'Models are still loading. Please wait...';
      return;
    }
    
    // Reset error and success states
    this.error = null;
    this.textError = null;
    this.artError = null;
    this.successMessage = null;
    
    // Set loading state and start progress tracking
    this.isGenerating = true;
    this.startProgressTracking();
    
    // Use the unified card generation service
    console.log('AppComponent: About to generate card with colors:', card.colors);
    console.log('AppComponent: Full card object:', card);
    this.cardService.generateCompleteCard(card)
      .pipe(
        catchError(error => {
          this.error = `Failed to generate card: ${error.message}`;
          console.error('Card generation error:', error);
          return of(null);
        }),
        finalize(() => {
          this.isGenerating = false;
          this.stopProgressTracking();
          // Reset the form's loading state
          if (this.cardFormComponent) {
            this.cardFormComponent.setGenerating(false);
          }
        })
      )
      .subscribe(generatedCard => {
        if (!generatedCard) {
          // Handle error case
          return;
        }
        
        // Update the current card with generated content
        this.currentCard = generatedCard;
        this.successMessage = 'Card generated successfully!';
      });
  }

  generateCardText(card: Card): void {
    // Reset error and success states
    this.textError = null;
    this.successMessage = null;
    
    // Set loading state
    this.isGeneratingText = true;
    
    // Generate only text content
    this.cardService.generateCardText(card)
      .pipe(
        catchError(error => {
          this.textError = `Failed to generate card text: ${error.message}`;
          console.error('Card text generation error:', error);
          return of(null);
        }),
        finalize(() => {
          this.isGeneratingText = false;
        })
      )
      .subscribe((updatedCard: Partial<Card> | null) => {
        if (!updatedCard) {
          // Handle error case
          return;
        }
        
        // Update the current card with generated text
        this.currentCard = { ...this.currentCard, ...updatedCard };
        this.successMessage = 'Card text generated successfully!';
      });
  }

  generateCardArt(card: Card): void {
    // Reset error and success states
    this.artError = null;
    this.successMessage = null;
    
    // Set loading state
    this.isGeneratingArt = true;
    
    // Generate only art content
    this.cardService.generateCardArt(card)
      .pipe(
        catchError(error => {
          this.artError = `Failed to generate card art: ${error.message}`;
          console.error('Card art generation error:', error);
          return of(null);
        }),
        finalize(() => {
          this.isGeneratingArt = false;
        })
      )
      .subscribe((updatedCard: Partial<Card> | null) => {
        if (!updatedCard) {
          // Handle error case
          return;
        }
        
        // Update the current card with generated art
        this.currentCard = { ...this.currentCard, ...updatedCard };
        this.successMessage = 'Card art generated successfully!';
      });
  }
  
  
  
  // Clear error messages
  clearError(): void {
    this.error = null;
    this.textError = null;
    this.artError = null;
  }
  
  // Clear success message
  clearSuccessMessage(): void {
    this.successMessage = null;
  }

  // Download artwork image
  downloadCardImage(): void {
    if (!this.currentCard.imageUrl) {
      console.warn('No image available to download');
      return;
    }

    try {
      // Create a temporary link element
      const link = document.createElement('a');
      link.href = this.currentCard.imageUrl;
      
      // Generate filename based on card name
      const cardName = this.currentCard.name || 'magic-card';
      const sanitizedName = cardName.replace(/[^a-z0-9]/gi, '_').toLowerCase();
      link.download = `${sanitizedName}_artwork.png`;
      
      // Temporarily add to DOM and trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      this.successMessage = 'Artwork downloaded successfully!';
      
      // Clear success message after 20 seconds
      setTimeout(() => {
        this.successMessage = null;
      }, 20000);
      
    } catch (error) {
      console.error('Error downloading image:', error);
      this.error = 'Failed to download artwork. Please try again.';
    }
  }

  // Download complete card image
  downloadCompleteCard(): void {
    if (!this.currentCard.cardImageUrl) {
      console.warn('No complete card image available to download');
      return;
    }

    try {
      // Create a temporary link element
      const link = document.createElement('a');
      link.href = this.currentCard.cardImageUrl;
      
      // Generate filename based on card name
      const cardName = this.currentCard.name || 'magic-card';
      const sanitizedName = cardName.replace(/[^a-z0-9]/gi, '_').toLowerCase();
      link.download = `${sanitizedName}_complete_card.png`;
      
      // Temporarily add to DOM and trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      this.successMessage = 'Complete card downloaded successfully!';
      
      // Clear success message after 20 seconds
      setTimeout(() => {
        this.successMessage = null;
      }, 20000);
      
    } catch (error) {
      console.error('Error downloading complete card:', error);
      this.error = 'Failed to download complete card. Please try again.';
    }
  }

  // Health check methods
  async checkModelsHealth(): Promise<void> {
    this.isCheckingHealth = true;
    this.healthError = null;
    
    try {
      console.log('🏥 Performing single health check...');
      
      // Perform a single health check
      const status = await this.healthService.performHealthCheck();
      
      this.healthStatus = status;
      this.modelsReady = status.status === 'healthy';
      this.isCheckingHealth = false;
      
      if (this.modelsReady) {
        console.log('✅ Models are ready!');
        this.showHealthBanner = false; // Hide banner immediately when ready
      } else {
        console.warn('⚠️ Models not ready:', status);
        this.healthError = status.message;
      }
      
    } catch (error: any) {
      console.error('❌ Health check failed:', error);
      this.isCheckingHealth = false;
      this.modelsReady = false;
      this.healthError = error.message || 'Health check failed';
    }
  }

  // Progress tracking methods
  startProgressTracking(): void {
    this.generationProgress = 0;
    this.generationStartTime = Date.now();
    this.showExtendedMessage = false;
    
    // Reset to first message
    this.currentMessage = this.generationMessages.length > 0 
      ? this.generationMessages[0] 
      : { title: 'Generating Your Magic Card', subtitle: 'Creating artwork and card text with AI...' };
    
    // Start message rotation every 5 seconds
    this.startMessageRotation();
    
    this.progressInterval = setInterval(() => {
      const elapsed = Date.now() - this.generationStartTime;
      const expectedDuration = 120000; // 2 minutes for warm runs
      const extendedThreshold = 180000; // 3 minutes threshold for cold start
      
      if (elapsed >= extendedThreshold && !this.showExtendedMessage) {
        this.showExtendedMessage = true;
        this.generationProgress = 95; // Show near completion but not 100%
      } else if (elapsed < expectedDuration) {
        // Progress smoothly from 0 to 90% over 2 minutes
        this.generationProgress = Math.min(90, (elapsed / expectedDuration) * 90);
      } else {
        // After 2 minutes, slowly approach 95% over the next minute
        this.generationProgress = Math.min(95, 90 + ((elapsed - expectedDuration) / 60000) * 5);
      }
    }, 100); // Update every 100ms for smooth progress
  }

  stopProgressTracking(): void {
    if (this.progressInterval) {
      clearInterval(this.progressInterval);
      this.progressInterval = null;
    }
    
    this.stopMessageRotation();
    
    this.generationProgress = 100;
    this.showExtendedMessage = false;
    
    // Reset progress after a short delay
    setTimeout(() => {
      this.generationProgress = 0;
    }, 500);
  }

  loadGenerationMessages(): void {
    this.http.get<{messages: any[]}>('/assets/generation-messages.json')
      .subscribe({
        next: (data) => {
          this.generationMessages = data.messages;
          console.log(`Loaded ${this.generationMessages.length} generation messages`);
        },
        error: (error) => {
          console.error('Failed to load generation messages:', error);
          // Fallback messages if file fails to load
          this.generationMessages = [
            { title: 'Generating Your Magic Card', subtitle: 'Creating artwork and card text with AI...' },
            { title: 'Shuffling the Digital Deck', subtitle: 'No mulligans needed here!' },
            { title: 'Tapping Mana Sources', subtitle: 'Converting coffee into card magic...' }
          ];
        }
      });
  }

  startMessageRotation(): void {
    if (this.generationMessages.length <= 1) return;
    
    let messageIndex = 0;
    this.messageInterval = setInterval(() => {
      // Don't rotate messages if we're showing extended message
      if (this.showExtendedMessage) return;
      
      // Fade out current message
      this.messageVisible = false;
      
      // After fade out completes, change message and fade back in
      setTimeout(() => {
        messageIndex = (messageIndex + 1) % this.generationMessages.length;
        this.currentMessage = this.generationMessages[messageIndex];
        this.messageVisible = true;
      }, 300); // 300ms to match CSS transition
      
    }, 5000); // Change message every 5 seconds
  }

  stopMessageRotation(): void {
    if (this.messageInterval) {
      clearInterval(this.messageInterval);
      this.messageInterval = null;
    }
  }

}
