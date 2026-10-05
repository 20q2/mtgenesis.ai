import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { 
  Card, 
  ColorOptions, 
  CardTypeOptions, 
  SupertypeOptions,
  CommonSubtypes, 
  RarityOptions, 
  Rarity, 
  ManaSymbols,
  createEmptyCard 
} from '../../models/card.model';
import { ManaService } from '../../services/mana.service';

@Component({
  selector: 'app-card-form',
  templateUrl: './card-form.component.html',
  styleUrls: ['./card-form.component.scss']
})
export class CardFormComponent implements OnInit {
  cardForm!: FormGroup;
  colorOptions = ColorOptions;
  supertypeOptions = SupertypeOptions;
  cardTypeOptions = CardTypeOptions;
  rarityOptions = RarityOptions;
  manaSymbols = ManaSymbols;
  commonSubtypes = CommonSubtypes;
  
  showPowerToughness = false;
  filteredSubtypes: string[] = [];
  isGenerating = false;
  hasGeneratedCard = false;
  lastGenerationTime: number | null = null;
  generationStartTime: number | null = null;
  
  @Input() modelsReady: boolean = false;
  /** Hide the Card Name field. */
  @Input() showName = true;
  /** Hide the Generate button (when the host page has its own). */
  @Input() showGenerate = true;
  @Output() cardChange = new EventEmitter<Card>();
  @Output() generateCard = new EventEmitter<Card>();
  @Output() regenerateText = new EventEmitter<Card>();

  constructor(private fb: FormBuilder, private manaService: ManaService) {}

  ngOnInit(): void {
    this.initForm();

    // Emit initial values
    this.onFormValueChanges();
    
    // Subscribe to form changes to emit updates
    this.cardForm.valueChanges.subscribe(() => {
      // Reset generated card flag when form becomes dirty
      if (this.cardForm.dirty && this.hasGeneratedCard) {
        this.hasGeneratedCard = false;
      }
      this.onFormValueChanges();
    });

    // Subscribe to type changes to show/hide power/toughness
    this.cardForm.get('type')?.valueChanges.subscribe(type => {
      this.updateTypeRelatedFields(type);
    });

    // Subscribe to subtype changes as well (for vehicles)
    this.cardForm.get('subtype')?.valueChanges.subscribe(subtype => {
      this.updatePowerToughnessVisibility();
    });
  }

  updateTypeRelatedFields(type: string): void {
    // Update available subtypes based on main type
    this.updateFilteredSubtypes(type);
    
    // Update power/toughness visibility (considering both type and subtype)
    this.updatePowerToughnessVisibility();
  }

  updatePowerToughnessVisibility(): void {
    const type = this.cardForm.get('type')?.value?.toLowerCase() || '';
    const subtype = this.cardForm.get('subtype')?.value?.toLowerCase() || '';
    
    // Show power/toughness for:
    // 1. Creatures (main type includes 'creature')
    // 2. Vehicles (subtype includes 'vehicle') - since they become creatures when crewed
    this.showPowerToughness = type.includes('creature') || subtype.includes('vehicle');
    
    console.log(`Power/Toughness visibility updated: type="${type}", subtype="${subtype}", show=${this.showPowerToughness}`);
    
    // Update validators as needed
    if (this.showPowerToughness) {
      this.cardForm.get('power')?.setValidators([]);
      this.cardForm.get('toughness')?.setValidators([]);
    } else {
      this.cardForm.get('power')?.clearValidators();
      this.cardForm.get('toughness')?.clearValidators();
    }
    this.cardForm.get('power')?.updateValueAndValidity();
    this.cardForm.get('toughness')?.updateValueAndValidity();
  }

  updateFilteredSubtypes(type: string): void {
    if (!type) {
      this.filteredSubtypes = [];
      return;
    }
    
    // Find which main type the selected type belongs to
    const mainType = Object.keys(this.commonSubtypes).find(key => 
      type.toLowerCase().includes(key.toLowerCase())
    );
    
    this.filteredSubtypes = mainType 
      ? this.commonSubtypes[mainType as keyof typeof this.commonSubtypes] 
      : [];
  }

  initForm(): void {
    const emptyCard = createEmptyCard();
    
    this.cardForm = this.fb.group({
      name: [emptyCard.name, [Validators.maxLength(30)]],
      manaCost: [emptyCard.manaCost],
      supertype: [emptyCard.supertype],
      type: [emptyCard.type, [Validators.maxLength(50)]],
      subtype: [emptyCard.subtype],
      colors: [emptyCard.colors],
      cmc: [emptyCard.cmc, [Validators.min(0)]],
      rarity: [emptyCard.rarity],
      description: [emptyCard.description],
      power: [emptyCard.power],
      toughness: [emptyCard.toughness],
      powerToughness: [''], // Combined field for power/toughness
      flavorText: [emptyCard.flavorText],
      setCode: [emptyCard.setCode],
      cardNumber: [emptyCard.cardNumber]
    });
  }

  onFormValueChanges(): void {
    const formValue = this.cardForm.value;
    
    // If the type doesn't include 'creature', remove power/toughness
    if (!this.showPowerToughness) {
      formValue.power = undefined;
      formValue.toughness = undefined;
    }
    
    // Auto-generate art prompt
    formValue.artPrompt = this.generateArtPromptText();
    
    console.log('CardFormComponent: Emitting card changes with colors:', formValue.colors);
    console.log('CardFormComponent: Auto-generated art prompt:', formValue.artPrompt);
    this.cardChange.emit(formValue as Card);
  }

  onSubmit(): void {
    if (!this.showGenerate) {
      return; // Enter in a field must not trigger generation when the host page owns the button
    }
    if (this.isGenerating) {
      return; // Prevent double submission
    }
    
    this.isGenerating = true;
    this.generationStartTime = Date.now(); // Track when generation started
    this.generateCard.emit(this.cardForm.value as Card);
  }

  // Method to reset loading state - should be called by parent component
  setGenerating(generating: boolean): void {
    this.isGenerating = generating;
    if (!generating) {
      // Card generation completed
      this.hasGeneratedCard = true;
      
      // Calculate generation time if we have a start time
      if (this.generationStartTime) {
        this.lastGenerationTime = (Date.now() - this.generationStartTime) / 1000; // Convert to seconds
        this.generationStartTime = null; // Reset start time
      }
    }
  }

  onRegenerateText(): void {
    if (this.isGenerating) {
      return; // Prevent action while generating
    }
    
    this.regenerateText.emit(this.cardForm.value as Card);
  }

  // Helper to check if regenerate button should be shown
  shouldShowRegenerateButton(): boolean {
    return false;
    return this.hasGeneratedCard && this.cardForm.pristine && !this.isGenerating;
  }

  isFieldInvalid(field: string): boolean {
    const control = this.cardForm.get(field);
    return !!control && control.invalid && (control.dirty || control.touched);
  }

  insertManaSymbol(symbol: string): void {
    const manaCostControl = this.cardForm.get('manaCost');
    if (manaCostControl) {
      const currentValue = manaCostControl.value || '';
      const newValue = currentValue + symbol;
      const reorderedValue = this.manaService.reorderManaSymbols(newValue);
      manaCostControl.setValue(reorderedValue);
      manaCostControl.markAsDirty();
      
      // Update colors and CMC after inserting mana symbol
      this.calculateCmcFromManaCost();
      this.updateColorsFromManaCost();
    }
  }

  clearManaCost(): void {
    this.cardForm.get('manaCost')?.setValue('');
    this.cardForm.get('manaCost')?.markAsDirty();
    
    // Update colors and CMC after clearing mana cost
    this.calculateCmcFromManaCost();
    this.updateColorsFromManaCost();
  }

  reorderManaCost(): void {
    const manaCostControl = this.cardForm.get('manaCost');
    if (manaCostControl) {
      const currentValue = manaCostControl.value || '';
      const reorderedValue = this.manaService.reorderManaSymbols(currentValue);
      if (reorderedValue !== currentValue) {
        manaCostControl.setValue(reorderedValue);
        manaCostControl.markAsDirty();
        
        // Update colors and CMC after reordering (just in case)
        this.calculateCmcFromManaCost();
        this.updateColorsFromManaCost();
      }
    }
  }

  onManaCostChange(): void {
    console.log('onManaCostChange called!');
    this.calculateCmcFromManaCost();
    this.updateColorsFromManaCost();
  }

  calculateCmcFromManaCost(): void {
    console.log('calculateCmcFromManaCost called');
    const manaCost = this.cardForm.get('manaCost')?.value || '';
    const cmc = this.manaService.calculateCMC(manaCost);
    
    // Update the CMC in the form if we have one
    if (this.cardForm.get('cmc')) {
      this.cardForm.get('cmc')?.setValue(cmc);
      this.cardForm.get('cmc')?.markAsDirty();
    }
  }


  updateColorsFromManaCost(): void {
    console.log('updateColorsFromManaCost called');
    const manaCost = this.cardForm.get('manaCost')?.value || '';
    const colorsFromMana = this.manaService.extractColorsFromManaCost(manaCost);
    
    console.log('Mana cost:', manaCost);
    console.log('Colors extracted from mana:', colorsFromMana);
    
    // Only use colors from mana cost, no merging
    this.cardForm.get('colors')?.setValue(colorsFromMana, { emitEvent: true });
    this.cardForm.get('colors')?.markAsDirty();
    
    console.log('Set colors to:', colorsFromMana);
    
    // Force emit the form changes immediately
    this.onFormValueChanges();
  }

  selectSupertype(supertype: string): void {
    this.cardForm.get('supertype')?.setValue(supertype);
    this.cardForm.get('supertype')?.markAsDirty();
  }

  selectCardType(type: string): void {
    this.cardForm.get('type')?.setValue(type);
    this.cardForm.get('type')?.markAsDirty();
  }

  selectSubtype(subtype: string): void {
    this.cardForm.get('subtype')?.setValue(subtype);
    this.cardForm.get('subtype')?.markAsDirty();
  }

  updateFullType(): void {
    const mainType = this.cardForm.get('type')?.value || '';
    const subtype = this.cardForm.get('subtype')?.value || '';
    
    if (mainType && subtype) {
      this.cardForm.get('type')?.setValue(`${mainType} — ${subtype}`);
    }
  }

  /** True when the named control holds exactly this value (drives the chips' pressed state). */
  isSelected(control: string, value: string): boolean {
    return this.cardForm.get(control)?.value === value;
  }

  setRarity(rarity: Rarity): void {
    this.cardForm.get('rarity')?.setValue(rarity);
    this.cardForm.get('rarity')?.markAsDirty();
  }

  /**
   * The art subject: name, type and a size hint. The server adds the painting style, color
   * mood and palette (proxy-server/image_generation.py), so they aren't repeated here; words
   * like "divine light" and "detailed digital art" made the art glossy and overexposed.
   */
  generateArtPromptText(): string {
    const name = this.cardForm.get('name')?.value || '';
    const supertype = this.cardForm.get('supertype')?.value || '';
    const type = this.cardForm.get('type')?.value || '';
    const subtype = this.cardForm.get('subtype')?.value || '';
    const isCreature = type.toLowerCase().includes('creature');

    // "Legendary Creature - Dragon" reads best to the image model as "a legendary dragon",
    // "Artifact - Equipment" as "an equipment artifact".
    const noun = subtype ? (isCreature ? subtype : `${subtype} ${type}`) : type;
    const typeDescription = [supertype, noun].filter(Boolean).join(' ').toLowerCase();

    const parts: string[] = [];
    if (name) parts.push(name);
    if (typeDescription) parts.push(`${/^[aeiou]/.test(typeDescription) ? 'an' : 'a'} ${typeDescription}`);
    if (isCreature) {
      const cmc = Number(this.cardForm.get('cmc')?.value) || 0;
      parts.push(this.creatureScale(cmc));
    }
    return parts.join(', ') || 'a fantasy scene';
  }

  private creatureScale(cmc: number): string {
    if (cmc <= 1) return 'small';
    if (cmc <= 2) return 'modest size';
    if (cmc <= 4) return 'medium scale';
    if (cmc <= 6) return 'large and imposing';
    if (cmc <= 8) return 'massive';
    return 'colossal';
  }

  onColorChange(event: any, color: string): void {
    const currentColors = this.cardForm.get('colors')?.value || [];
    let newColors: string[];
    
    if (event.target.checked) {
      // Add color if not already present
      if (!currentColors.includes(color)) {
        newColors = [...currentColors, color];
      } else {
        newColors = currentColors;
      }
    } else {
      // Remove color
      newColors = currentColors.filter((c: string) => c !== color);
    }
    
    // Update the form control and force emit
    this.cardForm.get('colors')?.setValue(newColors, { emitEvent: true });
    this.cardForm.get('colors')?.markAsDirty();
    
    console.log('Manual color change:', color, event.target.checked ? 'added' : 'removed');
    console.log('Updated colors array:', newColors);
    
    // Force emit the form changes immediately
    this.onFormValueChanges();
  }

  getColorName(colorValue: string): string {
    const colorOption = this.colorOptions.find(option => option.value === colorValue);
    return colorOption ? colorOption.label : colorValue;
  }
  
  // Method to handle combined power/toughness input
  onPowerToughnessChange(): void {
    const powerToughnessValue = this.cardForm.get('powerToughness')?.value || '';
    const parts = powerToughnessValue.split('/');
    
    if (parts.length === 2) {
      const power = parts[0].trim();
      const toughness = parts[1].trim();
      
      this.cardForm.get('power')?.setValue(power);
      this.cardForm.get('toughness')?.setValue(toughness);
    }
  }
  
  // Update combined field when individual fields change
  updatePowerToughnessDisplay(): void {
    const power = this.cardForm.get('power')?.value || '';
    const toughness = this.cardForm.get('toughness')?.value || '';
    
    if (power && toughness) {
      this.cardForm.get('powerToughness')?.setValue(`${power}/${toughness}`, { emitEvent: false });
    }
  }
  
  // Convert mana symbol to CSS class for mana font
  getSymbolClass(symbol: string): string {
    return this.manaService.getSymbolClass(symbol);
  }

  // Get formatted symbol data for buttons
  getFormattedSymbolForButton(manaSymbol: any) {
    return this.manaService.formatSymbolForButton(manaSymbol);
  }

  // Clear entire form and reset to empty card
  /**
   * After a card is generated: the fields the player left empty get what the AI chose for
   * them (the server's card director fills blanks; see proxy-server/card_fill.py), so they
   * can tweak and regenerate. Typed fields and the supertype are never touched.
   */
  fillBlanks(card: { name?: string; manaCost?: string; type?: string; subtype?: string;
                    power?: string; toughness?: string }): void {
    const blank = (key: string) => !String(this.cardForm.get(key)?.value ?? '').trim();
    const patch: Record<string, unknown> = {};
    for (const key of ['name', 'type', 'subtype'] as const) {
      if (blank(key) && card[key]) {
        patch[key] = card[key];
      }
    }
    const fillCost = blank('manaCost') && !!card.manaCost;
    if (fillCost) {
      patch['manaCost'] = card.manaCost;
    }
    if (blank('powerToughness') && card.power && card.toughness) {
      patch['power'] = card.power;
      patch['toughness'] = card.toughness;
      patch['powerToughness'] = `${card.power}/${card.toughness}`;
    }
    if (!Object.keys(patch).length) {
      return;
    }
    this.cardForm.patchValue(patch);
    if (fillCost) {
      this.onManaCostChange();  // colors and mana value follow the filled cost
    }
  }

  clearForm(): void {
    const emptyCard = createEmptyCard();
    this.cardForm.patchValue({
      name: emptyCard.name,
      manaCost: emptyCard.manaCost,
      supertype: emptyCard.supertype,
      type: emptyCard.type,
      subtype: emptyCard.subtype,
      colors: emptyCard.colors,
      cmc: emptyCard.cmc,
      rarity: emptyCard.rarity,
      description: emptyCard.description,
      power: emptyCard.power,
      toughness: emptyCard.toughness,
      powerToughness: '',
      flavorText: emptyCard.flavorText,
      setCode: emptyCard.setCode,
      cardNumber: emptyCard.cardNumber
    });
    
    // Mark the form as dirty to trigger updates
    this.cardForm.markAsDirty();
    
    // Reset UI state
    this.showPowerToughness = false;
    this.filteredSubtypes = [];

    // Emit the changes
    this.onFormValueChanges();
    
    console.log('Form cleared to empty state');
  }
}
